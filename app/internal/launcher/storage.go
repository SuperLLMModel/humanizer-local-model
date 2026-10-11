package launcher

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// 存储位置:两个目录,都能在「下载模型」那一页改。
//
//	运行目录(数据目录)设置、日志、更新包、更新结果都在这儿;
//	模型目录          GGUF 文件(.part 半截文件也在这儿)都在这儿。
//
// 两者默认都在系统盘(Windows %LOCALAPPDATA%\Humanizer),网页上可以改成别的盘。
// 改位置不会自动搬文件(几十 GB,又是系统自己做的决定),要搬得在网页上勾一下。

// modelsDir 模型文件放哪儿。locModelDir 为空就是数据目录下的 models/。
func (a *App) modelsDir() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.modelDirLocked()
}

// modelDirLocked 给已经拿着 a.mu 的地方用。
func (a *App) modelDirLocked() string {
	if a.locModelDir != "" {
		return a.locModelDir
	}
	return filepath.Join(a.dataDir, "models")
}

// modelPath 一个档位的本地文件。模型目录改了以后,这个跟着变。
func (a *App) modelPath(t Tier) string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return filepath.Join(a.modelDirLocked(), filepath.FromSlash(t.File))
}

// pathsNow 一次性取两个目录,免得夹在两次加锁中间读到一半换过的值。
func (a *App) pathsNow() (dataDir, modelsDir string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.dataDir, a.modelDirLocked()
}

// applyPaths 换存储位置。两个入参都空字符串 = 回到默认。
// move 为真时把已经下载好的模型文件搬到新位置(跨盘符时是复制,可能要几分钟)。
// 搬不过去(空间不够、写不进去)就整个作罢:已经挪过去的挪回来,位置不变,返回错误。
// 换完按新位置重新判断开机状态:该下的重新下,该起的重新起。
func (a *App) applyPaths(dataDir, modelDir string, move bool) (moved int, err error) {
	if !a.pathsMu.TryLock() {
		return 0, errors.New("上一次换位置还没做完,等它做完再试")
	}
	defer a.pathsMu.Unlock()
	if a.updates != nil && a.updates.busy() {
		return 0, errors.New("正在下载或安装更新,等更新做完(或暂停)再换位置")
	}
	curData, curModels := a.pathsNow()

	wantData := curData
	if s := strings.TrimSpace(dataDir); s != "" && !samePath(s, curData) {
		if a.dataDirLocked {
			return 0, errors.New("运行目录由启动参数 --data-dir 固定,改不了;要换就改启动参数")
		}
		p, err := cleanDir(s)
		if err != nil {
			return 0, err
		}
		if p, err = runDirFor(p); err != nil {
			return 0, err
		}
		if err := os.MkdirAll(filepath.Join(p, "logs"), 0o755); err != nil {
			return 0, fmt.Errorf("建不了日志目录:%v", err)
		}
		wantData = p
	}
	wantModels := ""
	if s := strings.TrimSpace(modelDir); s != "" {
		p, err := cleanDir(s)
		if err != nil {
			return 0, err
		}
		wantModels = p
	}
	if samePath(wantModels, filepath.Join(wantData, "models")) {
		wantModels = "" // 就是默认位置:记成"跟着运行目录走"
	}
	newModels := wantModels
	if newModels == "" {
		newModels = filepath.Join(wantData, "models")
	}

	// 外部服务模式:改写走外部服务,不碰本机模型文件,也没有自带引擎要停,
	// 所以不等进行中的改写、不把状态改成「启动中」;最后的 boot() 只会回到外部模式,不拉起自带引擎。
	a.mu.Lock()
	ext := a.externalOn()
	a.mu.Unlock()

	// 先停下来:引擎正开着模型文件,Windows 上不关掉就搬不动;下载协程要等它把半截文件关掉。
	// 手上正在改写的话等它写完,别把请求打断。
	for waited := time.Duration(0); !ext && a.inflight.Load() > 0 && waited < 30*time.Second; waited += 200 * time.Millisecond {
		time.Sleep(200 * time.Millisecond)
	}
	a.pauseDownload()
	a.waitDownloadStopped()
	a.stopEngine()
	a.mu.Lock()
	if a.phase == phaseReady && !ext { // 跨盘复制可能要几分钟,这期间别让网页以为还能改写
		a.phase = phaseStarting
	}
	a.mu.Unlock()
	// 停了就一定要重新判断开机状态,失败了也一样(否则引擎停着、页面还显示就绪)
	defer a.boot()
	if move && !samePath(curModels, newModels) {
		if moved, err = a.moveModelFiles(curModels, newModels); err != nil {
			return 0, err
		}
	}

	a.mu.Lock()
	a.locModelDir = wantModels
	a.dataDir = wantData
	s := a.settings
	a.mu.Unlock()

	if err := saveLocation(a.pointerDir, wantData, wantModels); err != nil {
		a.logf("记下存储位置失败:%v", err)
	}
	// 设置跟着运行目录走:新目录里也写一份,档位、下载源、端口、开关都还在。
	if err := saveSettings(a.settingsPath(), s); err != nil {
		return moved, fmt.Errorf("保存设置失败:%v", err)
	}
	if !samePath(curData, wantData) {
		if a.updates != nil {
			a.updates.setDataDir(wantData)
		}
		carryOverFile(filepath.Join(curData, "config.json"), filepath.Join(wantData, "config.json"))
		a.moveExternalKey(curData)
		_ = os.Remove(filepath.Join(curData, "instance.json"))
		a.logf("运行目录改到 %s,启动器日志从这里起写到新目录", wantData)
		a.switchLauncherLog(wantData)
	}
	a.writeInstance()
	a.logf("存储位置:运行目录 %s,模型目录 %s(搬了 %d 个文件)", wantData, newModels, moved)
	return moved, nil
}

// runDirFor 决定选中的文件夹能不能直接当运行目录。
// 运行目录里有 settings.json、config.json 这类通用名字的文件,启动时还会把 config.json 当配置读,
// 所以不能和用户自己的文件混在一起(会覆盖别人的 settings.json,读到别人的 config.json 就起不来)。
// 空文件夹、默认位置、以前用过的运行目录直接用;别的就在里面建一个 Humanizer 子文件夹。
func runDirFor(p string) (string, error) {
	if samePath(p, defaultDataDir()) || isRunDir(p) || isEmptyDir(p) {
		return p, nil
	}
	sub := filepath.Join(p, "Humanizer")
	if st, err := os.Stat(sub); err == nil && (!st.IsDir() || !(isRunDir(sub) || isEmptyDir(sub))) {
		return "", fmt.Errorf("%s 里已经有别的文件,里面的 Humanizer 也不是空的:请选一个空文件夹当运行目录", p)
	}
	return cleanDir(sub)
}

// isRunDir:Humanizer 用过的运行目录(每次启动都会写 logs/launcher.log)。
func isRunDir(p string) bool {
	return fileExists(filepath.Join(p, "logs", "launcher.log"))
}

// isEmptyDir:不算系统自己放的隐藏文件。
func isEmptyDir(p string) bool {
	ents, err := os.ReadDir(p)
	if err != nil {
		return false
	}
	for _, e := range ents {
		switch strings.ToLower(e.Name()) {
		case ".ds_store", "desktop.ini", "thumbs.db", ".humanizer-write-test":
		default:
			return false
		}
	}
	return true
}

// waitDownloadStopped 等下载协程退出(最多 30 秒),它退出前还会往半截文件里写。
func (a *App) waitDownloadStopped() {
	a.mu.Lock()
	done := a.dlDone
	a.mu.Unlock()
	if done == nil {
		return
	}
	select {
	case <-done:
	case <-time.After(30 * time.Second):
		a.logf("下载协程 30 秒还没退出,接着换位置")
	}
}

// switchLauncherLog 运行目录换了:launcher.log 换到新目录里接着写。
func (a *App) switchLauncherLog(dataDir string) {
	if a.logger == nil {
		return
	}
	f, err := os.OpenFile(filepath.Join(dataDir, "logs", "launcher.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		a.logf("新运行目录里开不了日志文件:%v", err)
		return
	}
	a.logger.SetOutput(io.MultiWriter(f, os.Stderr))
	if a.logFile != nil {
		_ = a.logFile.Close()
	}
	a.logFile = f
}

// modelFileSuffixes:一个档位在模型目录里可能有的文件(成品、下载半截、更新半截)。
var modelFileSuffixes = []string{"", ".part", ".part.json", ".update.part", ".update.part.json"}

// moveModelFiles 把 from 下的模型文件搬到 to。只搬配置里档位对应的文件,
// 用户自己放在同一个目录里的别的东西(比如别的 GGUF 模型)一概不碰。
// 新位置已经有同名文件就不动它。任何一个搬不过去就全部还原,返回错误。
// 跨盘符时 os.Rename 失败,改成复制:先复制成 .moving,全部成功后才改名、删旧文件。
func (a *App) moveModelFiles(from, to string) (int, error) {
	if from == "" || to == "" || samePath(from, to) {
		return 0, nil
	}
	type job struct{ src, dst string }
	var jobs []job
	for _, t := range a.cfg.Tiers {
		rel := filepath.FromSlash(t.File)
		for _, suf := range modelFileSuffixes {
			src, dst := filepath.Join(from, rel)+suf, filepath.Join(to, rel)+suf
			if fileExists(src) && !fileExists(dst) {
				jobs = append(jobs, job{src, dst})
			}
		}
	}
	if len(jobs) == 0 {
		return 0, nil
	}
	var renamed, copied []job
	undo := func() {
		for _, j := range renamed {
			_ = os.Rename(j.dst, j.src)
		}
		for _, j := range copied {
			_ = os.Remove(j.dst + ".moving")
		}
	}
	for _, j := range jobs {
		if err := os.MkdirAll(filepath.Dir(j.dst), 0o755); err != nil {
			undo()
			return 0, fmt.Errorf("建不了新模型目录:%v", err)
		}
		if os.Rename(j.src, j.dst) == nil {
			renamed = append(renamed, j)
			continue
		}
		st, err := os.Stat(j.src)
		if err != nil {
			undo()
			return 0, err
		}
		if free, ok := freeDiskBytes(filepath.Dir(j.dst)); ok && free < uint64(st.Size())+256<<20 {
			undo()
			return 0, fmt.Errorf("新位置空间不够:%s 要 %.1f GB,只剩 %.1f GB;位置没改", filepath.Base(j.src), float64(st.Size())/gib, float64(free)/gib)
		}
		a.logf("跨盘复制 %s → %s(%.1f GB)", j.src, j.dst, float64(st.Size())/gib)
		if err := cloneFile(j.src, j.dst+".moving"); err != nil {
			undo()
			return 0, fmt.Errorf("复制 %s 失败:%v;位置没改", filepath.Base(j.src), err)
		}
		copied = append(copied, j)
	}
	for _, j := range copied {
		if err := os.Rename(j.dst+".moving", j.dst); err != nil {
			a.logf("复制好的 %s 改名失败:%v", j.dst, err)
			continue
		}
		if err := os.Remove(j.src); err != nil {
			a.logf("删掉旧文件 %s 失败(新位置已经有了):%v", j.src, err)
		}
	}
	return len(jobs), nil
}

// moveExternalKey 运行目录换了:外部服务的 API key 跟着设置一起到新目录(settings.json 刚整份写过去,
// key 也以内存里这份为准),旧目录里那份删掉,免得密钥散落在两处。
// 没有 key 时把新目录里以前留下的旧 key 也删掉,和刚写过去的 settings.json 保持一致。
func (a *App) moveExternalKey(oldDataDir string) {
	a.mu.Lock()
	key := a.extKey
	a.mu.Unlock()
	if err := a.saveExternalKey(key); err != nil {
		a.logf("外部服务的 API key 写不到新运行目录:%v(旧目录里那份留着)", err)
		return
	}
	if old := filepath.Join(oldDataDir, externalKeyFile); !samePath(old, a.externalKeyPath()) {
		if err := os.Remove(old); err != nil && !errors.Is(err, os.ErrNotExist) {
			a.logf("删不掉旧运行目录里的 API key:%v", err)
		}
	}
}

// carryOverFile 把一个设置文件带到新目录(目标已经有了就不动)。
func carryOverFile(src, dst string) {
	if fileExists(dst) {
		return
	}
	b, err := os.ReadFile(src)
	if err != nil {
		return
	}
	_ = writeFileAtomic(dst, b)
}

// cloneFile 复制一个文件,修改时间也带过去(模型指纹缓存按大小+修改时间认文件,不带过去就得重算几分钟哈希)。
func cloneFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	st, err := in.Stat()
	if err != nil {
		return err
	}
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	_, err = io.Copy(out, in)
	if err == nil {
		err = out.Sync()
	}
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		_ = os.Remove(dst)
		return err
	}
	return os.Chtimes(dst, st.ModTime(), st.ModTime())
}

// ───────────────────────── HTTP ─────────────────────────

// handlePaths 保存网页上改的存储位置。
func (a *App) handlePaths(w http.ResponseWriter, r *http.Request) {
	var req struct {
		DataDir  string `json:"data_dir"`
		ModelDir string `json:"model_dir"`
		Move     bool   `json:"move"`
		Reset    bool   `json:"reset"` // 「恢复默认」:两个目录都回默认
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 8192)).Decode(&req); err != nil {
		writeJSON(w, 400, map[string]any{"error": "请求格式不对"})
		return
	}
	dataDir, modelDir := req.DataDir, req.ModelDir
	if req.Reset {
		// 恢复默认:模型目录回「运行目录下的 models」,运行目录回系统默认位置
		// (被 --data-dir 钉死时,那一半本来就动不了,别在这里报错)
		dataDir, modelDir = defaultDataDir(), ""
		if a.dataDirLocked {
			dataDir, _ = a.pathsNow()
		}
	}
	moved, err := a.applyPaths(dataDir, modelDir, req.Move)
	if err != nil {
		a.logf("改存储位置失败:%v", err)
		writeJSON(w, 400, map[string]any{"error": err.Error()})
		return
	}
	dataDir, modelsDir := a.pathsNow()
	writeJSON(w, 200, map[string]any{
		"ok": true, "moved": moved, "data_dir": dataDir, "model_dir": modelsDir,
	})
}

// handlePickDir 弹系统的「选文件夹」窗口。what=data|models;取消时返回空路径。
func (a *App) handlePickDir(w http.ResponseWriter, r *http.Request) {
	var req struct {
		What string `json:"what"`
	}
	_ = json.NewDecoder(http.MaxBytesReader(w, r.Body, 1024)).Decode(&req)
	what := "models"
	switch req.What {
	case "data":
		what = "data"
	case "models":
	default:
		writeJSON(w, 400, map[string]any{"error": "不知道要选哪个目录"})
		return
	}
	if what == "data" && a.dataDirLocked {
		writeJSON(w, 409, map[string]any{"error": "运行目录由启动参数 --data-dir 固定,改不了"})
		return
	}
	dataDir, modelsDir := a.pathsNow()
	from, title := modelsDir, "选择模型文件存放的文件夹"
	if what == "data" {
		from, title = dataDir, "选择运行目录(设置、日志、引擎更新)"
	}
	_ = os.MkdirAll(from, 0o755) // 选择器要一个已经存在的起始目录
	// 用户关掉网页不该让选择器留在屏幕上,所以不走请求的 context。
	// 两分钟上限:选择器要是压根没弹出来(用户目录被改坏等),也不能让按钮一直转着。
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	p, err := pickDir(ctx, from, title)
	if err != nil {
		a.logf("打开目录选择器失败:%v", err)
		msg := "打不开文件夹选择器:可以把路径直接粘到输入框里,再点「保存位置」"
		if errors.Is(err, context.DeadlineExceeded) {
			msg = "文件夹选择器两分钟没反应,已经关掉了:可以把路径直接粘到输入框里,再点「保存位置」"
		}
		writeJSON(w, 500, map[string]any{"error": msg})
		return
	}
	if p != "" {
		p, err = cleanDir(p)
		if err != nil {
			writeJSON(w, 400, map[string]any{"error": err.Error()})
			return
		}
	}
	writeJSON(w, 200, map[string]any{"path": p})
}
