package launcher

// 外部服务(0.3.4,HF 讨论 #11):用户已经在跑一个装着 humanizer 模型的 OpenAI 兼容服务(LM Studio、
// llama-server、Ollama、vLLM……),App 不再拉起自带的 llama.cpp,改写请求经启动器反代到那个服务的
// /chat/completions(流式)。网页还是只和 127.0.0.1 上的启动器说话(CSP、跨域、Host/Origin 检查都不变),
// 外部服务不用开 CORS;API key 只在启动器里加到请求头上,不进浏览器。
//
// 设置存在 settings.json 的 "external" 里;API key 单独存在数据目录的 external-api-key(权限 0600)。

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// ExternalSettings:settings.json 里的 "external"。
type ExternalSettings struct {
	Enabled bool   `json:"enabled"`
	URL     string `json:"url,omitempty"`   // 规范化后的基地址,比如 http://localhost:1234/v1
	Model   string `json:"model,omitempty"` // 请求里的 model;空 = 网页不带(服务端用当前加载的模型)
}

type externalStatus struct {
	Enabled bool   `json:"enabled"`
	URL     string `json:"url"`
	Model   string `json:"model"`
	HasKey  bool   `json:"has_key"`
}

const externalKeyFile = "external-api-key"

func (a *App) externalKeyPath() string { return filepath.Join(a.dataDir, externalKeyFile) }

func (a *App) loadExternalKey() string {
	b, err := os.ReadFile(a.externalKeyPath())
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}

func (a *App) saveExternalKey(key string) error {
	p := a.externalKeyPath()
	if key == "" {
		if err := os.Remove(p); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	tmp := p + ".tmp"
	if err := os.WriteFile(tmp, []byte(key), 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, p)
}

// externalOn:调用方持有 a.mu。
func (a *App) externalOn() bool {
	e := a.settings.External
	return e != nil && e.Enabled && e.URL != ""
}

func (a *App) externalStatusLocked() *externalStatus {
	s := &externalStatus{HasKey: a.extKey != ""}
	if e := a.settings.External; e != nil {
		s.Enabled, s.URL, s.Model = a.externalOn(), e.URL, e.Model
	}
	return s
}

// normalizeExternalURL 把用户填的地址整理成 OpenAI 兼容的基地址:
//
//	localhost:1234                          → http://localhost:1234/v1
//	http://localhost:1234/                  → http://localhost:1234/v1
//	http://127.0.0.1:1234/v1/chat/completions → http://127.0.0.1:1234/v1
//	http://host:11434/v1/                   → http://host:11434/v1
func normalizeExternalURL(raw string) (string, error) {
	s := strings.TrimSpace(raw)
	if s == "" {
		return "", errors.New("empty")
	}
	if !strings.Contains(s, "://") {
		s = "http://" + s
	}
	u, err := url.Parse(s)
	if err != nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") {
		return "", errors.New("bad_url")
	}
	if u.User != nil {
		return "", errors.New("bad_url") // 密码请填 API key,不要放进地址
	}
	p := strings.TrimRight(u.Path, "/")
	for _, suffix := range []string{"/chat/completions", "/completions", "/models"} {
		p = strings.TrimSuffix(p, suffix)
	}
	p = strings.TrimRight(p, "/")
	if p == "" {
		p = "/v1"
	}
	u.Path, u.RawPath, u.RawQuery, u.Fragment = p, "", "", ""
	return u.String(), nil
}

// 外部服务的连接:只限制建连和读响应头之前的部分;生成本身(长文、慢机器)不设总时限,用户可以随时停止。
var externalTransport = &http.Transport{
	Proxy:                 http.ProxyFromEnvironment, // localhost / 127.0.0.1 永远不走代理
	DialContext:           (&net.Dialer{Timeout: 5 * time.Second}).DialContext,
	TLSHandshakeTimeout:   10 * time.Second,
	MaxIdleConnsPerHost:   2,
	IdleConnTimeout:       60 * time.Second,
	ResponseHeaderTimeout: 0,
}

// externalProxy:/api/v1/chat/completions → <base>/chat/completions。
func (a *App) externalProxy(base *url.URL, key string) http.Handler {
	return &httputil.ReverseProxy{
		Rewrite: func(pr *httputil.ProxyRequest) {
			sub := strings.TrimPrefix(pr.In.URL.Path, "/api/v1")
			pr.Out.URL.Scheme = base.Scheme
			pr.Out.URL.Host = base.Host
			pr.Out.URL.Path = strings.TrimRight(base.Path, "/") + sub
			pr.Out.URL.RawPath = ""
			pr.Out.URL.RawQuery = ""
			pr.Out.Host = base.Host
			for _, h := range []string{"Origin", "Referer", "Cookie", "X-Humanizer", "Authorization"} {
				pr.Out.Header.Del(h)
			}
			if key != "" {
				pr.Out.Header.Set("Authorization", "Bearer "+key)
			}
		},
		Transport:     externalTransport,
		FlushInterval: -1,
		ModifyResponse: func(r *http.Response) error {
			// 外部服务给的 CORS / Cookie 头对同源网页没用,也别带回来
			for _, h := range []string{"Access-Control-Allow-Origin", "Access-Control-Allow-Credentials", "Set-Cookie"} {
				r.Header.Del(h)
			}
			return nil
		},
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			if r.Context().Err() != nil { // 用户点了停止
				return
			}
			writeJSON(w, http.StatusBadGateway, map[string]any{"error": map[string]any{
				"code": 502, "type": "external_unreachable", "url": base.String(),
				"message": "连不上外部服务 " + base.String() + ":" + err.Error(),
			}})
		},
	}
}

// probeExternal:GET <base>/models,确认服务在、key 对,顺便拿模型列表。
func probeExternal(ctx context.Context, base, key string) ([]string, int, error) {
	ctx, cancel := context.WithTimeout(ctx, 6*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "GET", strings.TrimRight(base, "/")+"/models", nil)
	if err != nil {
		return nil, 0, err
	}
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	resp, err := (&http.Client{Transport: externalTransport}).Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode != 200 {
		msg := strings.TrimSpace(string(body))
		var j struct {
			Error any `json:"error"`
		}
		if json.Unmarshal(body, &j) == nil && j.Error != nil {
			if m, ok := j.Error.(map[string]any); ok && m["message"] != nil {
				msg = fmt.Sprint(m["message"])
			} else {
				msg = fmt.Sprint(j.Error)
			}
		}
		if len(msg) > 300 {
			msg = msg[:300]
		}
		return nil, resp.StatusCode, fmt.Errorf("HTTP %d %s", resp.StatusCode, msg)
	}
	var list struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
		Models []struct { // Ollama 的 /api/tags 风格,有的代理会这样回
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.Unmarshal(body, &list); err != nil {
		return nil, resp.StatusCode, errors.New("not an OpenAI-compatible /models response")
	}
	var ids []string
	for _, m := range list.Data {
		if m.ID != "" {
			ids = append(ids, m.ID)
		}
	}
	for _, m := range list.Models {
		if m.Name != "" {
			ids = append(ids, m.Name)
		}
	}
	sort.Strings(ids)
	return ids, resp.StatusCode, nil
}

// enterExternal:停掉自带引擎、暂停下载(半截文件留着),直接就绪。
func (a *App) enterExternal() {
	a.stopEngine()
	a.pauseDownload()
	a.mu.Lock()
	a.phase = phaseReady
	a.errInfo = nil
	a.engInfo = engineInfo{Backend: "external", ReadyAt: time.Now()}
	if e := a.settings.External; e != nil {
		a.engInfo.Device = e.URL
	}
	a.mu.Unlock()
	a.writeInstance()
	a.logf("使用外部服务 %s,自带引擎不启动", a.engInfo.Device)
}

func (a *App) handleExternalGet(w http.ResponseWriter, r *http.Request) {
	a.mu.Lock()
	s := a.externalStatusLocked()
	a.mu.Unlock()
	writeJSON(w, 200, s)
}

// POST /app/external {enabled, url, model, api_key?}:api_key 不给 = 不改,给空串 = 清除。
func (a *App) handleExternalSet(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Enabled bool    `json:"enabled"`
		URL     string  `json:"url"`
		Model   string  `json:"model"`
		APIKey  *string `json:"api_key"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 8192)).Decode(&req); err != nil {
		writeJSON(w, 400, map[string]any{"error": "bad json", "code": "bad_json"})
		return
	}
	base := ""
	if strings.TrimSpace(req.URL) != "" || req.Enabled {
		u, err := normalizeExternalURL(req.URL)
		if err != nil {
			writeJSON(w, 400, map[string]any{"error": "bad url", "code": "bad_url"})
			return
		}
		base = u
	}
	model := strings.TrimSpace(req.Model)
	if len(model) > 256 {
		writeJSON(w, 400, map[string]any{"error": "model name too long", "code": "bad_model"})
		return
	}
	if req.APIKey != nil {
		key := strings.TrimSpace(*req.APIKey)
		if len(key) > 4096 || strings.ContainsAny(key, "\r\n") {
			writeJSON(w, 400, map[string]any{"error": "bad api key", "code": "bad_key"})
			return
		}
		if err := a.saveExternalKey(key); err != nil {
			writeJSON(w, 500, map[string]any{"error": err.Error(), "code": "save_failed"})
			return
		}
		a.mu.Lock()
		a.extKey = key
		a.mu.Unlock()
	}
	a.mu.Lock()
	was := a.externalOn()
	a.settings.External = &ExternalSettings{Enabled: req.Enabled, URL: base, Model: model}
	now := a.externalOn()
	s := a.settings
	a.restarts = 0
	a.mu.Unlock()
	if err := saveSettings(a.settingsPath(), s); err != nil {
		a.logf("保存设置失败:%v", err)
	}
	switch {
	case now:
		a.enterExternal() // 换了地址/模型也重新记一下
	case was:
		a.logf("关闭外部服务,恢复自带引擎")
		a.mu.Lock()
		a.engInfo = engineInfo{}
		a.mu.Unlock()
		a.boot()
	}
	a.mu.Lock()
	st := a.externalStatusLocked()
	a.mu.Unlock()
	writeJSON(w, 200, st)
}

// POST /app/external/test {url, api_key?}:api_key 不给就用已保存的。
func (a *App) handleExternalTest(w http.ResponseWriter, r *http.Request) {
	var req struct {
		URL    string  `json:"url"`
		APIKey *string `json:"api_key"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 8192)).Decode(&req); err != nil {
		writeJSON(w, 400, map[string]any{"ok": false, "code": "bad_json"})
		return
	}
	base, err := normalizeExternalURL(req.URL)
	if err != nil {
		writeJSON(w, 200, map[string]any{"ok": false, "code": "bad_url"})
		return
	}
	a.mu.Lock()
	key := a.extKey
	a.mu.Unlock()
	if req.APIKey != nil {
		key = strings.TrimSpace(*req.APIKey)
	}
	models, status, err := probeExternal(r.Context(), base, key)
	if err != nil {
		code := "unreachable"
		switch {
		case status == 401 || status == 403:
			code = "auth"
		case status != 0:
			code = "http"
		}
		writeJSON(w, 200, map[string]any{"ok": false, "code": code, "url": base, "message": err.Error()})
		return
	}
	writeJSON(w, 200, map[string]any{"ok": true, "url": base, "models": models})
}
