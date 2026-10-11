package launcher

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"testing/fstest"
)

func TestNormalizeExternalURL(t *testing.T) {
	ok := map[string]string{
		"http://localhost:1234/v1":                  "http://localhost:1234/v1",
		"  http://localhost:1234/v1/  ":             "http://localhost:1234/v1",
		"localhost:1234":                            "http://localhost:1234/v1",
		"http://127.0.0.1:1234/":                    "http://127.0.0.1:1234/v1",
		"http://127.0.0.1:1234/v1/chat/completions": "http://127.0.0.1:1234/v1",
		"http://127.0.0.1:8080/v1/completions":      "http://127.0.0.1:8080/v1",
		"http://localhost:11434/v1/models?x=1#y":    "http://localhost:11434/v1",
		"https://example.test/openai/v1":            "https://example.test/openai/v1",
		"http://[::1]:1234":                         "http://[::1]:1234/v1",
	}
	for in, want := range ok {
		got, err := normalizeExternalURL(in)
		if err != nil || got != want {
			t.Errorf("normalize(%q) = %q, %v; want %q", in, got, err, want)
		}
	}
	for _, bad := range []string{"", "   ", "ftp://host/v1", "http://", "http://user:pw@host/v1", "://x"} {
		if got, err := normalizeExternalURL(bad); err == nil {
			t.Errorf("normalize(%q) = %q, want error", bad, got)
		}
	}
}

// fakeOpenAI:最小的 OpenAI 兼容服务,记下收到的请求。
type fakeOpenAI struct {
	mu   sync.Mutex
	reqs []*http.Request
	body []map[string]any
	key  string
}

func (f *fakeOpenAI) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	var body map[string]any
	if r.Body != nil {
		_ = json.NewDecoder(r.Body).Decode(&body)
	}
	f.mu.Lock()
	f.reqs = append(f.reqs, r)
	f.body = append(f.body, body)
	f.mu.Unlock()
	if f.key != "" && r.Header.Get("Authorization") != "Bearer "+f.key {
		w.WriteHeader(401)
		io.WriteString(w, `{"error":{"message":"Invalid API key"}}`)
		return
	}
	switch r.URL.Path {
	case "/v1/models":
		io.WriteString(w, `{"object":"list","data":[{"id":"humanizer-12b-q8_0"},{"id":"another"}]}`)
	case "/v1/chat/completions":
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Access-Control-Allow-Origin", "*")
		for _, piece := range []string{"Hello", " there"} {
			fmt.Fprintf(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":%q}}]}\n\n", piece)
			w.(http.Flusher).Flush()
		}
		io.WriteString(w, "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	default:
		http.NotFound(w, r)
	}
}

// testLauncher:真的 App.handler(),只是没有引擎;dataDir 在临时目录。
func testLauncher(t *testing.T) (*App, string) {
	t.Helper()
	a := &App{cfg: testConfig(t), dataDir: t.TempDir(), web: fstest.MapFS{"index.html": {Data: []byte("hi")}}, apiKey: "k"}
	a.sys.RAMGB = 64
	a.rootCtx, a.shutdown = context.WithCancel(context.Background())
	t.Cleanup(a.shutdown)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	a.port = ln.Addr().(*net.TCPAddr).Port
	srv := &http.Server{Handler: a.handler()}
	go srv.Serve(ln)
	t.Cleanup(func() { srv.Close() })
	return a, fmt.Sprintf("http://127.0.0.1:%d", a.port)
}

func postApp(t *testing.T, base, path, body string) (int, map[string]any) {
	t.Helper()
	req, _ := http.NewRequest("POST", base+path, strings.NewReader(body))
	req.Header.Set("X-Humanizer", "1")
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&out)
	return resp.StatusCode, out
}

func getStatus(t *testing.T, base string) statusResp {
	t.Helper()
	resp, err := http.Get(base + "/app/status")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var s statusResp
	if err := json.NewDecoder(resp.Body).Decode(&s); err != nil {
		t.Fatal(err)
	}
	return s
}

func TestExternalServiceProxy(t *testing.T) {
	fake := &fakeOpenAI{key: "sk-test"}
	ext := httptest.NewServer(fake)
	defer ext.Close()
	a, base := testLauncher(t)
	a.boot()
	if st := getStatus(t, base); st.Phase != phaseSetup || st.External == nil || st.External.Enabled {
		t.Fatalf("没有模型、没开外部服务时应该在 setup:%+v", st)
	}

	// 测试连接:key 不对 → auth;对了 → 模型列表
	_, r := postApp(t, base, "/app/external/test", `{"url":"`+ext.URL+`","api_key":"wrong"}`)
	if r["ok"] != false || r["code"] != "auth" {
		t.Fatalf("错 key 应该报 auth:%v", r)
	}
	_, r = postApp(t, base, "/app/external/test", `{"url":"`+ext.URL+`/v1/","api_key":"sk-test"}`)
	if r["ok"] != true || fmt.Sprint(r["models"]) != "[another humanizer-12b-q8_0]" || r["url"] != ext.URL+"/v1" {
		t.Fatalf("测试连接:%v", r)
	}

	// 开启:直接就绪,不起引擎;key 存成 0600 文件,状态里只说有没有 key
	code, r := postApp(t, base, "/app/external", `{"enabled":true,"url":"`+ext.URL+`","model":"humanizer-12b-q8_0","api_key":"sk-test"}`)
	if code != 200 || r["enabled"] != true || r["has_key"] != true || r["url"] != ext.URL+"/v1" {
		t.Fatalf("开启外部服务:%d %v", code, r)
	}
	st := getStatus(t, base)
	if st.Phase != phaseReady || st.Engine == nil || st.Engine.Backend != "external" || !st.External.Enabled || st.External.Model != "humanizer-12b-q8_0" {
		t.Fatalf("开启后状态:%+v engine=%+v", st, st.Engine)
	}
	raw, _ := json.Marshal(st)
	if strings.Contains(string(raw), "sk-test") {
		t.Fatal("状态里不能带 API key")
	}
	fi, err := os.Stat(filepath.Join(a.dataDir, externalKeyFile))
	if err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS != "windows" && fi.Mode().Perm() != 0o600 {
		t.Fatalf("key 文件权限 %v,应为 0600", fi.Mode().Perm())
	}
	if s := loadSettings(a.settingsPath()); s.External == nil || !s.External.Enabled || s.External.URL != ext.URL+"/v1" {
		t.Fatalf("settings.json:%+v", s.External)
	}
	b, _ := os.ReadFile(a.settingsPath())
	if strings.Contains(string(b), "sk-test") {
		t.Fatal("settings.json 里不能有 API key")
	}

	// 反代:流式透传,带上 key,去掉浏览器的头
	req, _ := http.NewRequest("POST", base+"/api/v1/chat/completions", strings.NewReader(`{"model":"humanizer-12b-q8_0","messages":[{"role":"user","content":"draft"}],"stream":true}`))
	req.Header.Set("X-Humanizer", "1")
	req.Header.Set("Origin", base)
	req.Header.Set("Cookie", "a=b")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	var lines []string
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		if l := sc.Text(); l != "" {
			lines = append(lines, l)
		}
	}
	resp.Body.Close()
	if resp.StatusCode != 200 || len(lines) != 4 || lines[3] != "data: [DONE]" || !strings.Contains(lines[0], "Hello") {
		t.Fatalf("流式透传:%d %q", resp.StatusCode, lines)
	}
	if resp.Header.Get("Access-Control-Allow-Origin") != "" {
		t.Fatal("外部服务的 CORS 头不该带回网页")
	}
	fake.mu.Lock()
	last := fake.reqs[len(fake.reqs)-1]
	lastBody := fake.body[len(fake.body)-1]
	fake.mu.Unlock()
	if last.URL.Path != "/v1/chat/completions" || last.Header.Get("Authorization") != "Bearer sk-test" ||
		last.Header.Get("Origin") != "" || last.Header.Get("Cookie") != "" || last.Header.Get("X-Humanizer") != "" {
		t.Fatalf("转发的请求不对:%s %v", last.URL.Path, last.Header)
	}
	if lastBody["model"] != "humanizer-12b-q8_0" {
		t.Fatalf("请求体要原样转发:%v", lastBody)
	}

	// 外部模式下只放行 chat/completions;没带 X-Humanizer 的 POST 照样被挡
	for _, p := range []string{"/api/tokenize", "/api/completion", "/api/v1/models", "/api/v1/completions"} {
		req, _ := http.NewRequest("POST", base+p, strings.NewReader(`{}`))
		req.Header.Set("X-Humanizer", "1")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != 404 {
			t.Errorf("外部模式 %s 应该 404,得到 %d", p, resp.StatusCode)
		}
	}
	if resp, _ := http.Post(base+"/api/v1/chat/completions", "application/json", strings.NewReader(`{}`)); resp.StatusCode != 403 {
		t.Errorf("没带 X-Humanizer 应该 403,得到 %d", resp.StatusCode)
	}

	// 外部模式下不能换档 / 重启自带引擎;startEngine 也不会把自带引擎拉起来
	if code, _ := postApp(t, base, "/app/setup", `{"tier":"q8"}`); code != 409 {
		t.Errorf("外部模式下 /app/setup 应该 409,得到 %d", code)
	}
	if code, _ := postApp(t, base, "/app/engine/restart", `{}`); code != 409 {
		t.Errorf("外部模式下 /app/engine/restart 应该 409,得到 %d", code)
	}
	q8, _ := a.cfg.tier("q8")
	a.startEngine(q8)
	if st := getStatus(t, base); st.Phase != phaseReady || st.Engine.Backend != "external" {
		t.Fatalf("外部模式下 startEngine 不该改状态:%+v", st)
	}
	if (appEngine{a}).usesTier("q8") {
		t.Fatal("外部模式下模型更新不该认为引擎在用这个档位")
	}

	// 不传 api_key = 不改;传空串 = 清除
	postApp(t, base, "/app/external", `{"enabled":true,"url":"`+ext.URL+`","model":""}`)
	if a.loadExternalKey() != "sk-test" {
		t.Fatal("没传 api_key 时应该保留原来的 key")
	}
	postApp(t, base, "/app/external", `{"enabled":true,"url":"`+ext.URL+`","api_key":""}`)
	if _, err := os.Stat(filepath.Join(a.dataDir, externalKeyFile)); !os.IsNotExist(err) {
		t.Fatal("api_key 传空串应该删掉 key 文件")
	}

	// 服务挂了:清楚的 502
	ext.Close()
	req, _ = http.NewRequest("POST", base+"/api/v1/chat/completions", strings.NewReader(`{}`))
	req.Header.Set("X-Humanizer", "1")
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	var e struct {
		Error struct {
			Type, URL, Message string
		}
	}
	json.NewDecoder(resp.Body).Decode(&e)
	resp.Body.Close()
	if resp.StatusCode != 502 || e.Error.Type != "external_unreachable" || e.Error.URL == "" {
		t.Fatalf("服务挂了应该 502 external_unreachable:%d %+v", resp.StatusCode, e)
	}
	_, r = postApp(t, base, "/app/external/test", `{"url":"`+ext.URL+`"}`)
	if r["ok"] != false || r["code"] != "unreachable" {
		t.Fatalf("测试连接到挂掉的服务:%v", r)
	}

	// 关闭:恢复原来的行为(这里没有模型 → 回到 setup),反代回到 llama 白名单
	code, r = postApp(t, base, "/app/external", `{"enabled":false,"url":"`+ext.URL+`"}`)
	if code != 200 || r["enabled"] != false {
		t.Fatalf("关闭:%d %v", code, r)
	}
	if st := getStatus(t, base); st.Phase != phaseSetup || st.External.Enabled || st.External.URL != ext.URL+"/v1" {
		t.Fatalf("关闭后应回到 setup 且记住地址:%+v %+v", st.Phase, st.External)
	}
	req, _ = http.NewRequest("POST", base+"/api/v1/chat/completions", strings.NewReader(`{}`))
	req.Header.Set("X-Humanizer", "1")
	resp, _ = http.DefaultClient.Do(req)
	resp.Body.Close()
	if resp.StatusCode != 404 {
		t.Fatalf("关闭外部服务后 /api/v1/chat/completions 应该 404,得到 %d", resp.StatusCode)
	}
	if code, _ := postApp(t, base, "/app/external", `{"enabled":true,"url":"not a url at all://"}`); code != 400 {
		t.Fatalf("坏地址应该 400,得到 %d", code)
	}
}

// 重启后:settings.json 里开着外部服务,就不起引擎、直接就绪。
func TestExternalServiceBoot(t *testing.T) {
	a, base := testLauncher(t)
	a.settings.External = &ExternalSettings{Enabled: true, URL: "http://127.0.0.1:9/v1"}
	a.boot()
	st := getStatus(t, base)
	if st.Phase != phaseReady || st.Engine == nil || st.Engine.Backend != "external" || st.Engine.Device != "http://127.0.0.1:9/v1" {
		t.Fatalf("开机直接走外部服务:%+v %+v", st.Phase, st.Engine)
	}
	a.mu.Lock()
	eng := a.engine
	a.mu.Unlock()
	if eng != nil {
		t.Fatal("外部服务模式下不该有自带引擎进程")
	}
}
