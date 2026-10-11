package launcher

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func testApp(t *testing.T) *App {
	t.Helper()
	dir := t.TempDir()
	a := &App{cfg: testConfig(t), dataDir: dir, rootCtx: context.Background(), opts: Options{Version: "0.3.2"}}
	a.settings = loadSettings(a.settingsPath())
	return a
}

func TestModelsDirDefault(t *testing.T) {
	a := testApp(t)
	if got, want := a.modelsDir(), filepath.Join(a.dataDir, "models"); got != want {
		t.Fatalf("默认模型目录 %s ≠ %s", got, want)
	}
	a.locModelDir = filepath.Join(t.TempDir(), "D-models")
	if got := a.modelsDir(); got != a.locModelDir {
		t.Fatalf("改过以后应跟着变,实际 %s", got)
	}
	// modelPath 要跟着模型目录走(引擎加载的正是这个路径)
	tier, _ := a.cfg.tier("q8")
	if !strings.HasPrefix(a.modelPath(tier), a.locModelDir) {
		t.Fatalf("modelPath %s 不在 %s 下面", a.modelPath(tier), a.locModelDir)
	}
}

// fakeHome 把默认数据目录指到临时目录,测试别去读写本机真的 App 数据。
func fakeHome(t *testing.T) string {
	t.Helper()
	tmp := t.TempDir()
	t.Setenv("LOCALAPPDATA", tmp)
	t.Setenv("XDG_DATA_HOME", tmp)
	t.Setenv("HOME", tmp)
	return tmp
}

func TestResolveStartPathsDefaults(t *testing.T) {
	fakeHome(t)
	sp := resolveStartPaths("")
	if sp.DataDir != defaultDataDir() || sp.DataDirLock || sp.PointerDir != defaultDataDir() || sp.ModelDir != "" {
		t.Fatalf("没设过就该是默认位置:%+v", sp)
	}
}

func TestResolveStartPathsLocked(t *testing.T) {
	home := fakeHome(t)
	// 默认目录里有一份指针:给了 --data-dir 就完全不该看它
	elsewhere := filepath.Join(home, "elsewhere")
	if err := saveLocation(defaultDataDir(), elsewhere, filepath.Join(home, "m-default")); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	sp := resolveStartPaths(dir)
	if !sp.DataDirLock || !samePath(sp.PointerDir, dir) || sp.ModelDir != "" {
		t.Fatalf("显式指定的运行目录应锁死、指针记在它自己里面、不读默认目录的指针:%+v", sp)
	}
	if !samePath(sp.DataDir, dir) {
		t.Fatalf("运行目录应是 %s,实际 %s", dir, sp.DataDir)
	}
	// 钉死时网页上改的模型目录记在钉死的目录里,重启还在
	models := filepath.Join(home, "m-locked")
	if err := saveLocation(dir, dir, models); err != nil {
		t.Fatal(err)
	}
	if sp := resolveStartPaths(dir); !samePath(sp.ModelDir, models) || !samePath(sp.DataDir, dir) {
		t.Fatalf("钉死时模型目录应能找回:%+v", sp)
	}
	if loc, _ := readLocation(dir); loc.DataDir != "" {
		t.Fatalf("钉死时不该记运行目录:%+v", loc)
	}
	// 回到默认模型目录就删掉指针
	if err := saveLocation(dir, dir, ""); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, pointerName)); !os.IsNotExist(err) {
		t.Fatal("模型目录回默认后应删掉指针")
	}
}

func TestLocationJSONPlainUTF8(t *testing.T) {
	for _, p := range []string{
		`D:\模型 文件夹\R&D <新>`,
		"/Volumes/外置 盘/Humanizer \"引号\" 😀",
		`\\nas\共享\models`,
	} {
		b := locationJSON(location{DataDir: p, ModelDir: p + "2"})
		// 卸载程序只认 \\ 和 \" 两种转义:中文、&、<、> 都得原样写
		if strings.Contains(string(b), `\u`) {
			t.Fatalf("不该有 \\u 转义:%s", b)
		}
		var got location
		if err := json.Unmarshal(b, &got); err != nil {
			t.Fatalf("写出来的不是合法 JSON:%v\n%s", err, b)
		}
		if got.DataDir != p || got.ModelDir != p+"2" {
			t.Fatalf("读回来不一样:%q %q", got.DataDir, got.ModelDir)
		}
	}
	if b := locationJSON(location{ModelDir: "/m"}); strings.Contains(string(b), "data_dir") {
		t.Fatalf("空字段不该写出来:%s", b)
	}
}

func TestResolveStartPathsIgnoresRelative(t *testing.T) {
	home := fakeHome(t)
	os.MkdirAll(defaultDataDir(), 0o755)
	os.WriteFile(filepath.Join(defaultDataDir(), pointerName), []byte(`{"data_dir":"rel/dir","model_dir":"../x"}`), 0o644)
	sp := resolveStartPaths("")
	if !samePath(sp.DataDir, defaultDataDir()) || sp.ModelDir != "" {
		t.Fatalf("相对路径(被手改坏了)应当没写:%+v", sp)
	}
	if _, err := os.Stat(filepath.Join(home, "rel")); err == nil {
		t.Fatal("不该照着相对路径建目录")
	}
}

func TestSaveLocationRoundTrip(t *testing.T) {
	tmp := fakeHome(t)
	data2, models2 := filepath.Join(tmp, "D", "Humanizer"), filepath.Join(tmp, "D", "gguf")
	if err := saveLocation(defaultDataDir(), data2, models2); err != nil {
		t.Fatal(err)
	}
	sp := resolveStartPaths("")
	if !samePath(sp.DataDir, data2) || !samePath(sp.ModelDir, models2) {
		t.Fatalf("重启后没找回位置:%+v", sp)
	}
	if sp.DataDirLock || sp.PointerDir != defaultDataDir() {
		t.Fatalf("网页上改的应可再改:%+v", sp)
	}
	// 都回默认时把指针删掉,免得留下一个指向旧位置的空壳
	if err := saveLocation(defaultDataDir(), defaultDataDir(), ""); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(tmp, pointerName)); !os.IsNotExist(err) {
		t.Fatalf("回到默认后应删掉 %s", pointerName)
	}
}

func TestSaveLocationIgnoresBadPointer(t *testing.T) {
	tmp := fakeHome(t)
	// 指到一个建不出来的目录(是文件,不是目录):当没设过,回默认
	file := filepath.Join(tmp, "a-file")
	os.WriteFile(file, []byte("x"), 0o644)
	os.WriteFile(filepath.Join(tmp, pointerName), []byte(`{"data_dir":`+strconvQuote(file)+`}`), 0o644)
	if sp := resolveStartPaths(""); !samePath(sp.DataDir, defaultDataDir()) {
		t.Fatalf("坏指针应被忽略,实际 %s", sp.DataDir)
	}
	os.WriteFile(filepath.Join(tmp, pointerName), []byte(`{不是 json`), 0o644)
	if sp := resolveStartPaths(""); !samePath(sp.DataDir, defaultDataDir()) {
		t.Fatalf("坏 JSON 应被忽略,实际 %s", sp.DataDir)
	}
}

func TestCleanDir(t *testing.T) {
	tmp := t.TempDir()
	got, err := cleanDir(filepath.Join(tmp, "new", "nested"))
	if err != nil {
		t.Fatal(err)
	}
	if got != filepath.Join(tmp, "new", "nested") {
		t.Fatalf("路径没被收拾干净:%s", got)
	}
	if st, err := os.Stat(got); err != nil || !st.IsDir() {
		t.Fatalf("目录应该已经建出来:%v", err)
	}
	file := filepath.Join(tmp, "f.txt")
	os.WriteFile(file, nil, 0o644)
	if _, err := cleanDir(file); err == nil {
		t.Fatal("指向一个文件应该报错")
	}
	if _, err := cleanDir("  "); err == nil {
		t.Fatal("空路径应该报错")
	}
}

func TestApplyPathsMovesModels(t *testing.T) {
	a := testApp(t)
	old := filepath.Join(a.dataDir, "models")
	os.MkdirAll(old, 0o755)
	model := filepath.Join(old, "humanizer-12b-Q8_0.gguf")
	os.WriteFile(model, []byte("GGUF fake"), 0o644)

	newModels := filepath.Join(t.TempDir(), "gguf")
	moved, err := a.applyPaths("", newModels, true)
	if err != nil {
		t.Fatal(err)
	}
	if moved != 1 {
		t.Fatalf("应搬 1 个文件,实际 %d", moved)
	}
	if !fileExists(filepath.Join(newModels, "humanizer-12b-Q8_0.gguf")) {
		t.Fatal("文件没搬到新位置")
	}
	if fileExists(model) {
		t.Fatal("旧位置的文件应该已经不在了")
	}
	if a.modelsDir() != newModels {
		t.Fatalf("模型目录应是 %s,实际 %s", newModels, a.modelsDir())
	}
}

func TestApplyPathsWithoutMoveKeepsFiles(t *testing.T) {
	a := testApp(t)
	old := filepath.Join(a.dataDir, "models")
	os.MkdirAll(old, 0o755)
	model := filepath.Join(old, "humanizer-12b-Q8_0.gguf")
	os.WriteFile(model, []byte("GGUF fake"), 0o644)

	moved, err := a.applyPaths("", filepath.Join(t.TempDir(), "gguf"), false)
	if err != nil {
		t.Fatal(err)
	}
	if moved != 0 || !fileExists(model) {
		t.Fatalf("不勾「一起搬」就不该动文件(moved=%d, 旧文件在=%v)", moved, fileExists(model))
	}
}

func TestApplyPathsMovesPartFile(t *testing.T) {
	a := testApp(t)
	old := filepath.Join(a.dataDir, "models")
	os.MkdirAll(old, 0o755)
	// 下了一半的文件也得带走,不然续传要重新下几十 GB
	os.WriteFile(old+filepath.FromSlash("/humanizer-12b-Q8_0.gguf.part"), []byte("half"), 0o644)

	newModels := filepath.Join(t.TempDir(), "gguf")
	if moved, err := a.applyPaths("", newModels, true); err != nil || moved != 1 {
		t.Fatalf("半截文件应被搬走:%d %v", moved, err)
	}
	if !fileExists(filepath.Join(newModels, "humanizer-12b-Q8_0.gguf.part")) {
		t.Fatal("半截文件没在新位置")
	}
}

func TestApplyPathsBlockedWhenLocked(t *testing.T) {
	tmp := fakeHome(t)
	a := testApp(t)
	a.dataDirLocked = true
	if _, err := a.applyPaths(filepath.Join(tmp, "other"), "", false); err == nil {
		t.Fatal("运行目录被 --data-dir 固定时不该接受改动")
	}
	// 模型目录还是能改的
	if _, err := a.applyPaths("", filepath.Join(tmp, "m"), false); err != nil {
		t.Fatalf("模型目录应该还能改:%v", err)
	}
}

func TestApplyPathsResetsToDefault(t *testing.T) {
	tmp := fakeHome(t)
	a := testApp(t)
	a.pointerDir = defaultDataDir()
	if _, err := a.applyPaths(filepath.Join(tmp, "elsewhere"), filepath.Join(tmp, "m"), false); err != nil {
		t.Fatal(err)
	}
	if !samePath(a.modelsDir(), filepath.Join(tmp, "m")) {
		t.Fatalf("模型目录没改过来:%s", a.modelsDir())
	}
	if _, err := a.applyPaths("", "", false); err != nil {
		t.Fatal(err)
	}
	if !samePath(a.modelsDir(), filepath.Join(a.dataDir, "models")) {
		t.Fatalf("恢复默认后应是 %s,实际 %s", filepath.Join(a.dataDir, "models"), a.modelsDir())
	}
	if sp := resolveStartPaths(""); sp.ModelDir != "" {
		t.Fatalf("指针里不该再留模型目录:%+v", sp)
	}
}

func TestApplyPathsCarriesSettings(t *testing.T) {
	tmp := fakeHome(t)
	a := testApp(t)
	a.pointerDir = defaultDataDir()
	a.settings.Tier = "q8"
	a.settings.Endpoint = "mirror"
	if err := saveSettings(a.settingsPath(), a.settings); err != nil {
		t.Fatal(err)
	}
	next := filepath.Join(tmp, "next")
	if _, err := a.applyPaths(next, "", false); err != nil {
		t.Fatal(err)
	}
	if !samePath(a.dataDir, next) {
		t.Fatalf("运行目录应是 %s,实际 %s", next, a.dataDir)
	}
	// 新目录里要有设置(档位、下载源都在),否则下次启动要重选
	got := loadSettings(a.settingsPath())
	if got.Tier != "q8" || got.Endpoint != "mirror" {
		t.Fatalf("设置没带过去:%+v", got)
	}
	if _, err := os.Stat(filepath.Join(next, "logs")); err != nil {
		t.Fatalf("新运行目录里应该有 logs 子目录:%v", err)
	}
	// 指针得让下次启动找得到
	if sp := resolveStartPaths(""); !samePath(sp.DataDir, next) {
		t.Fatalf("重启后没找回新运行目录:%+v", sp)
	}
}

func TestMoveModelFilesSkipsExisting(t *testing.T) {
	a := testApp(t)
	from, to := filepath.Join(t.TempDir(), "from"), filepath.Join(t.TempDir(), "to")
	os.MkdirAll(from, 0o755)
	os.MkdirAll(to, 0o755)
	q8 := "humanizer-12b-Q8_0.gguf"
	os.WriteFile(filepath.Join(from, q8), []byte("old"), 0o644)
	os.WriteFile(filepath.Join(to, q8), []byte("already here"), 0o644)
	if n, err := a.moveModelFiles(from, to); n != 0 || err != nil {
		t.Fatalf("新位置已有的文件不该被覆盖,却搬了 %d 个(%v)", n, err)
	}
	b, _ := os.ReadFile(filepath.Join(to, q8))
	if string(b) != "already here" {
		t.Fatal("新位置的文件被覆盖了")
	}
}

// 模型目录可能是用户自己的模型库:只搬我们档位的文件,别的 GGUF 和杂物一概不碰。
func TestMoveModelFilesOnlyOurs(t *testing.T) {
	a := testApp(t)
	from, to := filepath.Join(t.TempDir(), "from"), filepath.Join(t.TempDir(), "to")
	os.MkdirAll(filepath.Join(from, "sub"), 0o755)
	ours := []string{"humanizer-12b-Q8_0.gguf", "humanizer-12b-Q4_K_M.gguf.part", "humanizer-12b-Q4_K_M.gguf.part.json"}
	theirs := []string{"llama-3-8b.Q4.gguf", "notes.txt", "humanizer-12b-Q8_0.gguf.bak", filepath.Join("sub", "humanizer-12b-Q8_0.gguf")}
	for _, n := range append(append([]string{}, ours...), theirs...) {
		os.WriteFile(filepath.Join(from, n), []byte(n), 0o644)
	}
	n, err := a.moveModelFiles(from, to)
	if err != nil || n != len(ours) {
		t.Fatalf("应只搬 %d 个,实际 %d(%v)", len(ours), n, err)
	}
	for _, f := range ours {
		if !fileExists(filepath.Join(to, f)) || fileExists(filepath.Join(from, f)) {
			t.Fatalf("%s 没搬过去", f)
		}
	}
	for _, f := range theirs {
		if !fileExists(filepath.Join(from, f)) || fileExists(filepath.Join(to, f)) {
			t.Fatalf("%s 不是我们的文件,不该动", f)
		}
	}
}

// 中途搬不动:已经挪过去的挪回来,位置不变。
func TestApplyPathsMoveFailureRollsBack(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Windows 上只读目录权限不靠 chmod")
	}
	if os.Geteuid() == 0 {
		t.Skip("root 不受目录权限限制")
	}
	a := testApp(t)
	a.cfg.Tiers = append(a.cfg.Tiers, Tier{ID: "sub", File: "lite/humanizer-lite.gguf"})
	old := filepath.Join(a.dataDir, "models")
	os.MkdirAll(filepath.Join(old, "lite"), 0o755)
	q8 := filepath.Join(old, "humanizer-12b-Q8_0.gguf")
	os.WriteFile(q8, []byte("GGUF q8"), 0o644)
	os.WriteFile(filepath.Join(old, "lite", "humanizer-lite.gguf"), []byte("GGUF lite"), 0o644)
	to := filepath.Join(t.TempDir(), "gguf")
	os.MkdirAll(filepath.Join(to, "lite"), 0o755)
	os.Chmod(filepath.Join(to, "lite"), 0o555) // 第二个文件放不进去
	t.Cleanup(func() { os.Chmod(filepath.Join(to, "lite"), 0o755) })

	if _, err := a.applyPaths("", to, true); err == nil {
		t.Fatal("搬不过去应该报错")
	}
	if !fileExists(q8) || fileExists(filepath.Join(to, "humanizer-12b-Q8_0.gguf")) {
		t.Fatal("已经挪过去的文件应该挪回原处")
	}
	if !samePath(a.modelsDir(), old) {
		t.Fatalf("搬失败了位置不该变,实际 %s", a.modelsDir())
	}
}

func TestCloneFileKeepsModTime(t *testing.T) {
	dir := t.TempDir()
	src, dst := filepath.Join(dir, "a"), filepath.Join(dir, "b")
	os.WriteFile(src, []byte("data"), 0o644)
	when := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	os.Chtimes(src, when, when)
	if err := cloneFile(src, dst); err != nil {
		t.Fatal(err)
	}
	st, _ := os.Stat(dst)
	if !st.ModTime().Equal(when) {
		t.Fatalf("修改时间没带过去:%v", st.ModTime())
	}
}

func TestHandlePaths(t *testing.T) {
	a := testApp(t)
	mux := http.NewServeMux()
	mux.HandleFunc("POST /app/paths", a.handlePaths)
	do := func(body string) (int, map[string]any) {
		req := httptest.NewRequest("POST", "/app/paths", strings.NewReader(body))
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, req)
		var m map[string]any
		json.Unmarshal(rec.Body.Bytes(), &m)
		return rec.Code, m
	}
	newModels := filepath.Join(t.TempDir(), "gguf")
	code, m := do(`{"model_dir":` + strconvQuote(newModels) + `}`)
	if code != 200 || m["ok"] != true {
		t.Fatalf("保存模型目录:%d %v", code, m)
	}
	if got := m["model_dir"]; got != newModels {
		t.Fatalf("返回的模型目录 %v ≠ %s", got, newModels)
	}
	// 指向一个文件:报 400,位置保持不变
	file := filepath.Join(t.TempDir(), "f.txt")
	os.WriteFile(file, nil, 0o644)
	if code, m := do(`{"model_dir":` + strconvQuote(file) + `}`); code != 400 || m["error"] == nil {
		t.Fatalf("指向文件应报 400:%d %v", code, m)
	}
	if a.modelsDir() != newModels {
		t.Fatalf("失败后不该改位置,实际 %s", a.modelsDir())
	}
}

func TestHandlePathsReset(t *testing.T) {
	tmp := fakeHome(t)
	a := testApp(t)
	a.pointerDir = defaultDataDir()
	if _, err := a.applyPaths("", filepath.Join(tmp, "m"), false); err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /app/paths", a.handlePaths)
	req := httptest.NewRequest("POST", "/app/paths", strings.NewReader(`{"reset":true}`))
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("恢复默认:%d %s", rec.Code, rec.Body.String())
	}
	if !samePath(a.modelsDir(), filepath.Join(a.dataDir, "models")) {
		t.Fatalf("恢复默认后模型目录应是 %s,实际 %s", filepath.Join(a.dataDir, "models"), a.modelsDir())
	}
	// 数据目录是钉死的,恢复默认时那一半别动
	a2 := testApp(t)
	a2.dataDirLocked = true
	mux2 := http.NewServeMux()
	mux2.HandleFunc("POST /app/paths", a2.handlePaths)
	before, _ := a2.pathsNow()
	req2 := httptest.NewRequest("POST", "/app/paths", strings.NewReader(`{"reset":true}`))
	rec2 := httptest.NewRecorder()
	mux2.ServeHTTP(rec2, req2)
	if rec2.Code != 200 {
		t.Fatalf("钉死时恢复默认不该报错:%d %s", rec2.Code, rec2.Body.String())
	}
	if after, _ := a2.pathsNow(); !samePath(before, after) {
		t.Fatalf("钉死时运行目录不该变:%s → %s", before, after)
	}
}

func TestHandlePickDirRejectsUnknown(t *testing.T) {
	a := testApp(t)
	mux := http.NewServeMux()
	mux.HandleFunc("POST /app/pick-dir", a.handlePickDir)
	req := httptest.NewRequest("POST", "/app/pick-dir", strings.NewReader(`{"what":"nope"}`))
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != 400 {
		t.Fatalf("未知的目标应报 400,实际 %d", rec.Code)
	}
}

func TestHandlePickDirLocked(t *testing.T) {
	a := testApp(t)
	a.dataDirLocked = true
	mux := http.NewServeMux()
	mux.HandleFunc("POST /app/pick-dir", a.handlePickDir)
	req := httptest.NewRequest("POST", "/app/pick-dir", strings.NewReader(`{"what":"data"}`))
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != 409 {
		t.Fatalf("被钉死时应报 409,实际 %d", rec.Code)
	}
}

func strconvQuote(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

// 选一个已经有别的文件的文件夹当运行目录:不能把 settings.json / config.json 混进去,改成里面的 Humanizer 子文件夹。
func TestApplyPathsRunDirNeverMixesWithUserFiles(t *testing.T) {
	tmp := fakeHome(t)
	a := testApp(t)
	a.pointerDir = defaultDataDir()

	busy := filepath.Join(tmp, "Documents")
	os.MkdirAll(busy, 0o755)
	os.WriteFile(filepath.Join(busy, "config.json"), []byte(`{"not":"ours"}`), 0o644)
	os.WriteFile(filepath.Join(busy, "settings.json"), []byte("mine"), 0o644)
	if _, err := a.applyPaths(busy, "", false); err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join(busy, "Humanizer"); !samePath(a.dataDir, want) {
		t.Fatalf("有别的文件的文件夹应改用子文件夹 %s,实际 %s", want, a.dataDir)
	}
	if b, _ := os.ReadFile(filepath.Join(busy, "settings.json")); string(b) != "mine" {
		t.Fatal("用户自己的 settings.json 被覆盖了")
	}

	// 空文件夹、以前用过的运行目录:直接用
	empty := filepath.Join(tmp, "empty")
	os.MkdirAll(empty, 0o755)
	os.WriteFile(filepath.Join(empty, ".DS_Store"), nil, 0o644)
	if _, err := a.applyPaths(empty, "", false); err != nil || !samePath(a.dataDir, empty) {
		t.Fatalf("空文件夹应直接用:%s %v", a.dataDir, err)
	}
	used := filepath.Join(tmp, "used")
	os.MkdirAll(filepath.Join(used, "logs"), 0o755)
	os.WriteFile(filepath.Join(used, "logs", "launcher.log"), nil, 0o644)
	os.WriteFile(filepath.Join(used, "settings.json"), []byte("{}"), 0o644)
	if _, err := a.applyPaths(used, "", false); err != nil || !samePath(a.dataDir, used) {
		t.Fatalf("以前用过的运行目录应直接用:%s %v", a.dataDir, err)
	}

	// 里面的 Humanizer 也被占了:报错,位置不变
	taken := filepath.Join(tmp, "taken")
	os.MkdirAll(filepath.Join(taken, "Humanizer"), 0o755)
	os.WriteFile(filepath.Join(taken, "a.txt"), nil, 0o644)
	os.WriteFile(filepath.Join(taken, "Humanizer", "README.md"), nil, 0o644)
	if _, err := a.applyPaths(taken, "", false); err == nil {
		t.Fatal("子文件夹也被占了应报错")
	}
	if !samePath(a.dataDir, used) {
		t.Fatalf("报错时位置不该变:%s", a.dataDir)
	}
}
