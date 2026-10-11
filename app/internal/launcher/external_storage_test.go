package launcher

// 外部服务(0.3.4)和存储位置(PR #9)两个功能碰在一起的情况。

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"testing"
)

// 外部服务开着时改运行目录 + 搬模型:自带引擎不能被拉起来;外部服务的设置和 API key 跟着运行目录走,
// 旧目录里不留 key;重启(从新目录读)后还是外部模式、key 还在。
func TestApplyPathsInExternalMode(t *testing.T) {
	tmp := fakeHome(t)
	a, base := testLauncher(t)
	a.pointerDir = defaultDataDir()
	tier := a.cfg.Tiers[0]
	a.settings.Tier = tier.ID
	model := a.modelPath(tier)
	if err := os.MkdirAll(filepath.Dir(model), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(model, []byte("GGUF"), 0o644); err != nil { // 有模型文件:不是外部模式的话 boot 会起引擎
		t.Fatal(err)
	}
	a.settings.External = &ExternalSettings{Enabled: true, URL: "http://127.0.0.1:9/v1", Model: "humanizer-12b-q8_0"}
	if err := a.saveExternalKey("sk-move"); err != nil {
		t.Fatal(err)
	}
	a.extKey = "sk-move"
	if err := saveSettings(a.settingsPath(), a.settings); err != nil {
		t.Fatal(err)
	}
	a.boot()
	old := a.dataDir

	next := filepath.Join(tmp, "next run")
	code, out := postApp(t, base, "/app/paths", `{"data_dir":`+strconv.Quote(next)+`,"move":true}`)
	if code != 200 || out["moved"] != float64(1) {
		t.Fatalf("改位置:%d %v", code, out)
	}
	a.mu.Lock()
	eng, engCancel := a.engine, a.engineCancel
	a.mu.Unlock()
	if eng != nil || engCancel != nil {
		t.Fatal("外部服务模式下改位置把自带引擎拉起来了")
	}
	st := getStatus(t, base)
	if st.Phase != phaseReady || st.Engine == nil || st.Engine.Backend != "external" || st.External == nil || !st.External.Enabled || !st.External.HasKey {
		t.Fatalf("改完位置应该还是外部模式就绪:%s %+v %+v", st.Phase, st.Engine, st.External)
	}
	if !samePath(st.DataDir, next) || !fileExists(filepath.Join(next, "models", filepath.FromSlash(tier.File))) {
		t.Fatalf("运行目录/模型没到新位置:%s", st.DataDir)
	}
	newKey := filepath.Join(next, externalKeyFile)
	b, err := os.ReadFile(newKey)
	if err != nil || string(b) != "sk-move" {
		t.Fatalf("API key 没跟到新运行目录:%q %v", b, err)
	}
	if runtime.GOOS != "windows" {
		if fi, _ := os.Stat(newKey); fi.Mode().Perm() != 0o600 {
			t.Fatalf("新目录里的 key 权限应是 0600,是 %v", fi.Mode().Perm())
		}
	}
	if _, err := os.Stat(filepath.Join(old, externalKeyFile)); !os.IsNotExist(err) {
		t.Fatalf("旧运行目录里还留着 API key:%v", err)
	}
	got := loadSettings(filepath.Join(next, "settings.json"))
	if got.External == nil || !got.External.Enabled || got.External.URL != "http://127.0.0.1:9/v1" || got.External.Model != "humanizer-12b-q8_0" {
		t.Fatalf("外部服务设置没跟到新运行目录:%+v", got.External)
	}
	// 「重启」:从新运行目录读 key
	b2 := &App{dataDir: next}
	if k := b2.loadExternalKey(); k != "sk-move" {
		t.Fatalf("重启后读不到 key:%q", k)
	}
	// 外部模式下换档仍然 409(改位置没把这条规则弄丢)
	if code, _ := postApp(t, base, "/app/setup", `{"tier":"`+tier.ID+`"}`); code != 409 {
		t.Fatalf("外部模式下换档应该 409,得到 %d", code)
	}
}

// 没有 key 时换运行目录:新目录里以前留下的旧 key 删掉(和刚写过去的 settings.json 一致)。
func TestApplyPathsDropsStaleKeyInNewRunDir(t *testing.T) {
	tmp := fakeHome(t)
	a := testApp(t)
	a.pointerDir = defaultDataDir()
	next := filepath.Join(tmp, "old-run")
	if err := os.MkdirAll(filepath.Join(next, "logs"), 0o755); err != nil {
		t.Fatal(err)
	}
	_ = os.WriteFile(filepath.Join(next, "logs", "launcher.log"), nil, 0o644) // 以前用过的运行目录
	_ = os.WriteFile(filepath.Join(next, externalKeyFile), []byte("sk-stale"), 0o600)
	if _, err := a.applyPaths(next, "", false); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(next, externalKeyFile)); !os.IsNotExist(err) {
		t.Fatalf("新运行目录里的旧 key 应该删掉:%v", err)
	}
}

// 正在换存储位置时,开关外部服务返回 409,不会在搬模型的半路拉起引擎。
func TestExternalSetBusyWhilePathsChange(t *testing.T) {
	a, base := testLauncher(t)
	a.pathsMu.Lock()
	code, out := postApp(t, base, "/app/external", `{"enabled":true,"url":"http://127.0.0.1:9/v1"}`)
	a.pathsMu.Unlock()
	if code != 409 || out["code"] != "paths_busy" {
		t.Fatalf("换位置期间开外部服务应该 409 paths_busy:%d %v", code, out)
	}
	a.mu.Lock()
	on := a.externalOn()
	a.mu.Unlock()
	if on {
		t.Fatal("409 了却还是把外部服务打开了")
	}
	if code, _ := postApp(t, base, "/app/external", `{"enabled":true,"url":"http://127.0.0.1:9/v1"}`); code != 200 {
		t.Fatalf("换完位置后应该能开:%d", code)
	}
}

// 外部模式的改写请求也算「正在改写」:App 更新要等它写完。
func TestInflightCountsExternalChat(t *testing.T) {
	a := &App{}
	seen := -1
	h := a.countInflight(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { seen = int(a.inflight.Load()) }))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/api/v1/chat/completions", nil))
	if seen != 1 || a.inflight.Load() != 0 {
		t.Fatalf("外部服务的改写请求没计数:%d", seen)
	}
}
