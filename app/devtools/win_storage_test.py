"""Windows 上装一遍、改存储位置、卸一遍,核对「只删我们自己的东西」。

只在 GitHub 的 Windows runner(一次性虚拟机)上跑:它会装真的安装包、动真的 %LOCALAPPDATA%\\Humanizer、
在 D 盘建目录。别在自己的电脑上跑。

    python devtools/win_storage_test.py dist/Humanizer-0.0.0-windows-x64-setup.exe

安装包里的引擎是假引擎(devtools/fakellama),不加载真模型。
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

SETUP = os.path.abspath(sys.argv[1])
LAD = os.environ["LOCALAPPDATA"]
DEF = os.path.join(LAD, "Humanizer")
INST = os.path.join(LAD, "Programs", "Humanizer")
HOME = os.environ["USERPROFILE"]
PORT = 47830
Q8 = "humanizer-12b-Q8_0.gguf"
ROOT = "D:\\hz-ci"  # 这次测试在 D 盘建的东西都在这下面

RUN_DIR = ROOT + "\\中文 目录\\运行 目录"
MODEL_DIR = ROOT + "\\模型 库 (用户自己的)"

passed, failed = 0, []


def check(name, cond):
    global passed
    if cond:
        passed += 1
        print("  ✓", name, flush=True)
    else:
        failed.append(name)
        print("  ✗", name, flush=True)


def write(path, data=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data if isinstance(data, bytes) else data.encode("utf-8"))


def exists(*p):
    return os.path.exists(os.path.join(*p))


def junction(link, target):
    os.makedirs(target, exist_ok=True)
    os.makedirs(os.path.dirname(link), exist_ok=True)
    subprocess.run(["cmd", "/c", "mklink", "/J", link, target], check=True, capture_output=True)


def procs(prefix):
    out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, errors="replace").stdout
    return [l for l in out.splitlines() if l.lower().startswith('"' + prefix.lower())]


def install():
    subprocess.run([SETUP, "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"], check=True, timeout=600)
    assert exists(INST, "Humanizer.exe"), "安装后找不到 Humanizer.exe"


def uninstall(delete_data):
    env = dict(os.environ)
    env.pop("HUMANIZER_UNINSTALL_DELETE_DATA", None)
    if delete_data:
        env["HUMANIZER_UNINSTALL_DELETE_DATA"] = "1"
    log = os.path.join(os.environ.get("RUNNER_TEMP", LAD), "uninstall.log")
    subprocess.run([os.path.join(INST, "unins000.exe"), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/LOG=" + log],
                   env=env, check=True, timeout=300)
    # 卸载程序会把自己复制到临时目录再跑(_iu*.tmp),原进程可能先退出:等它真的做完
    for _ in range(240):
        if not exists(INST, "unins000.exe") and not procs("_iu"):
            break
        time.sleep(0.5)
    time.sleep(1)
    assert not exists(INST, "Humanizer.exe"), "卸载后程序文件还在"
    try:  # 卸载脚本自己写的两行(读到的 location.json、决定清哪些目录)
        for line in open(log, encoding="utf-8", errors="replace"):
            if "location.json:" in line or "要清理" in line:
                print("    卸载日志:", line.strip().split(" ", 2)[-1])
    except OSError:
        pass


def api(path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", method="POST" if body is not None else "GET",
                                 data=json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None,
                                 headers={"X-Humanizer": "1", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def start():
    env = dict(os.environ, HUMANIZER_UPDATE_API="http://127.0.0.1:9", HUMANIZER_UPDATE_DELAY_SEC="3600")
    p = subprocess.Popen([os.path.join(INST, "Humanizer.exe"), "--no-browser", "--port", str(PORT), "--idle-exit", "0"], env=env)
    for _ in range(100):
        try:
            api("/app/status")
            return p
        except OSError:
            time.sleep(0.2)
    raise SystemExit("App 没起来")


def wait_phase(phase, sec=40):
    for _ in range(sec * 5):
        if api("/app/status")[1].get("phase") == phase:
            return True
        time.sleep(0.2)
    print("    当前状态:", api("/app/status")[1].get("phase"), api("/app/status")[1].get("error"))
    return False


def quit_app(p):
    try:
        api("/app/quit", {})
    except OSError:
        pass
    p.wait(timeout=30)


def same(a, b):
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def reset_world():
    for d in (DEF, ROOT):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(ROOT, exist_ok=True)


print("1. 没改过位置、静默卸载:和以前一样,数据留着")
reset_world()
install()
write(os.path.join(DEF, "models", Q8), b"GGUF")
uninstall(delete_data=False)
check("静默卸载(没要求删数据)时默认目录原样留着", exists(DEF, "models", Q8))

print("2. 装上,在网页接口里把运行目录、模型目录改到 D 盘带中文和空格的目录,勾「一起搬」")
reset_world()
install()
write(os.path.join(DEF, "models", Q8), b"GGUF")
write(os.path.join(MODEL_DIR, "llama-3-8b.Q4_K_M.gguf"), b"not ours")
write(os.path.join(MODEL_DIR, "notes.txt"), b"not ours")
app = start()
check("用默认位置起来、找到模型、就绪", wait_phase("ready"))
st = api("/app/status")[1]
check("没改过时 data_dir 是默认目录", same(st["data_dir"], DEF) and not st["data_dir_locked"])
code, r = api("/app/paths", {"data_dir": RUN_DIR, "model_dir": MODEL_DIR, "move": True})
check(f"保存位置成功(HTTP {code} {r})", code == 200 and r.get("moved") == 1)
check("引擎用新位置的模型重新就绪", wait_phase("ready"))
st = api("/app/status")[1]
check("状态里的运行目录是中文路径", same(st["data_dir"], RUN_DIR))
check("状态里的模型目录是中文路径", same(st["model_dir"], MODEL_DIR))
check("模型搬到了新位置,旧位置没了", exists(MODEL_DIR, Q8) and not exists(DEF, "models", Q8))
check("用户自己的文件没被搬、没被动", exists(MODEL_DIR, "llama-3-8b.Q4_K_M.gguf") and exists(MODEL_DIR, "notes.txt"))
check("设置写进了新运行目录", exists(RUN_DIR, "settings.json"))
check("引擎日志写在新运行目录", exists(RUN_DIR, "logs", "llama-server.log"))
check("启动器日志也换到了新运行目录", exists(RUN_DIR, "logs", "launcher.log"))
raw = open(os.path.join(DEF, "location.json"), "rb").read()
print("    location.json:", raw.decode("utf-8", "replace").replace("\n", " "))
check("location.json 里中文原样写、没有 \\u 转义(卸载程序只认 \\\\ 和 \\\")", b"\\u" not in raw)
loc = json.loads(raw)
check("location.json 读回来就是那两个目录", same(loc["data_dir"], RUN_DIR) and same(loc["model_dir"], MODEL_DIR))

# 选一个已经有别的东西的文件夹当运行目录:要改用里面的 Humanizer 子文件夹,不能覆盖别人的 settings.json
busy = ROOT + "\\已经有东西"
write(os.path.join(busy, "settings.json"), b"user's own")
write(os.path.join(busy, "config.json"), b"{not json")
code, r = api("/app/paths", {"data_dir": busy, "model_dir": MODEL_DIR})
st = api("/app/status")[1]
check("有别的文件的文件夹 → 用里面的 Humanizer 子文件夹", code == 200 and same(st["data_dir"], busy + "\\Humanizer"))
check("别人的 settings.json 没被覆盖", open(os.path.join(busy, "settings.json"), "rb").read() == b"user's own")
code, r = api("/app/paths", {"data_dir": RUN_DIR, "model_dir": MODEL_DIR})
check("改回中文运行目录", code == 200 and wait_phase("ready"))

# 很长的路径(超过 260 个字符)
longdir = ROOT + "\\" + "\\".join(["很长的目录名" * 6] * 8)
assert len(longdir) > 300
code, r = api("/app/paths", {"data_dir": RUN_DIR, "model_dir": longdir, "move": True})
check(f"超过 MAX_PATH 的模型目录(长度 {len(longdir)})能设、模型搬得进去(HTTP {code} {r.get('moved')})", code == 200 and r.get("moved") == 1)
check("长路径下的模型能被(假)引擎加载", wait_phase("ready"))
code, r = api("/app/paths", {"data_dir": RUN_DIR, "model_dir": MODEL_DIR, "move": True})
check("搬回来,引擎就绪", code == 200 and r.get("moved") == 1 and wait_phase("ready"))

# 不能写的位置:盘符不存在
code, r = api("/app/paths", {"data_dir": "Q:\\nope", "model_dir": MODEL_DIR})
check(f"不存在的盘报错、位置不变(HTTP {code})", code == 400 and same(api("/app/status")[1]["data_dir"], RUN_DIR))
quit_app(app)

print("3. 重启、覆盖安装(自动更新走的就是覆盖安装):位置和数据都还在")
app = start()
check("重启后按 location.json 找回位置并就绪", wait_phase("ready"))
st = api("/app/status")[1]
check("重启后运行目录、模型目录都对", same(st["data_dir"], RUN_DIR) and same(st["model_dir"], MODEL_DIR))
quit_app(app)
install()
check("覆盖安装没动数据", exists(DEF, "location.json") and exists(MODEL_DIR, Q8) and exists(RUN_DIR, "settings.json"))
app = start()
check("覆盖安装后照样按改过的位置就绪", wait_phase("ready") and same(api("/app/status")[1]["model_dir"], MODEL_DIR))
quit_app(app)

print("4. 卸载并删数据:默认目录整个删,改过的目录只删我们的文件;链接指向的东西不碰")
write(os.path.join(RUN_DIR, "我的笔记.txt"), b"mine")              # 用户后来放进运行目录的
write(os.path.join(RUN_DIR, "updates", "Humanizer-9.9.9-windows-x64-setup.exe"), b"pkg")
victim1, victim2 = ROOT + "\\victim1", ROOT + "\\victim2"
write(os.path.join(victim1, "keep.txt"))
write(os.path.join(victim2, "keep.txt"))
junction(os.path.join(DEF, "link-to-victim"), victim1)
junction(os.path.join(RUN_DIR, "updates", "humanizer-updater-x"), victim2)
uninstall(delete_data=True)
check("默认目录删掉了", not exists(DEF))
check("默认目录里的目录联接指向的东西还在", exists(victim1, "keep.txt"))
check("运行目录里我们的文件删了", not exists(RUN_DIR, "settings.json") and not exists(RUN_DIR, "logs"))
check("运行目录里下载的新版安装包删了", not exists(RUN_DIR, "updates", "Humanizer-9.9.9-windows-x64-setup.exe"))
check("运行目录里名字像更新助手的目录联接只删了链接,指向的东西还在", exists(victim2, "keep.txt"))
check("运行目录里用户自己的文件还在(所以运行目录本身也留着)", exists(RUN_DIR, "我的笔记.txt"))
check("模型目录里我们的模型删了", not exists(MODEL_DIR, Q8))
check("模型目录里用户自己的模型和文件还在", exists(MODEL_DIR, "llama-3-8b.Q4_K_M.gguf") and exists(MODEL_DIR, "notes.txt"))

print("5. 改过、但运行目录里只有我们的东西:卸载后整个文件夹都没了")
reset_world()
install()
clean_run = ROOT + "\\纯净 运行目录"
write(os.path.join(DEF, "models", Q8), b"GGUF")
app = start()
wait_phase("ready")
code, _ = api("/app/paths", {"data_dir": clean_run, "model_dir": "", "move": True})
check("改运行目录、模型跟着运行目录走(models 子目录)", code == 200 and wait_phase("ready") and exists(clean_run, "models", Q8))
quit_app(app)
uninstall(delete_data=True)
check("只有我们文件的运行目录整个删掉了", not exists(clean_run))

print("6. location.json 被改坏、被篡改:一律不碰用户的东西")
cases = []
# 指到用户目录,还伪造了标记文件
cases.append(("运行目录指到用户目录(伪造了 logs\\launcher.log)", {"data_dir": HOME},
              [os.path.join(HOME, "settings.json"), os.path.join(HOME, "logs", "launcher.log")]))
cases.append(("运行目录指到盘符根", {"data_dir": "D:\\"},
              ["D:\\settings.json", "D:\\logs\\launcher.log"]))
cases.append(("运行目录带 ..", {"data_dir": ROOT + "\\a\\..\\victim3"},
              [ROOT + "\\victim3\\settings.json", ROOT + "\\victim3\\logs\\launcher.log"]))
cases.append(("运行目录用正斜杠", {"data_dir": ROOT.replace("\\", "/") + "/victim4"},
              [ROOT + "\\victim4\\settings.json", ROOT + "\\victim4\\logs\\launcher.log"]))
cases.append(("运行目录是相对路径", {"data_dir": "victim5"},
              [os.path.join(INST, "victim5", "settings.json")]))
cases.append(("运行目录是文档文件夹", {"data_dir": os.path.join(HOME, "Documents")},
              [os.path.join(HOME, "Documents", "settings.json"), os.path.join(HOME, "Documents", "logs", "launcher.log")]))
cases.append(("运行目录没有 launcher.log(不是我们用过的)", {"data_dir": ROOT + "\\victim6"},
              [ROOT + "\\victim6\\settings.json", ROOT + "\\victim6\\config.json"]))
cases.append(("模型目录指到盘符根", {"model_dir": "D:\\"}, ["D:\\" + Q8]))
cases.append(("模型目录是共享根", {"model_dir": "\\\\localhost\\D$"}, ["D:\\" + Q8]))
for name, loc, victims in cases:
    reset_world()
    install()
    for v in victims:
        write(v, b"victim")
    write(os.path.join(DEF, "location.json"), json.dumps(loc))
    uninstall(delete_data=True)
    check(name + ":用户的文件都还在", all(os.path.exists(v) for v in victims))
    check(name + ":默认目录照常删掉", not exists(DEF))
    for v in victims:
        try:
            os.remove(v)
        except OSError:
            pass
    shutil.rmtree(os.path.join(HOME, "logs"), ignore_errors=True)
    shutil.rmtree(os.path.join(HOME, "Documents", "logs"), ignore_errors=True)
    shutil.rmtree("D:\\logs", ignore_errors=True)

for name, content in [("不是 JSON", b"{not json"), ("用了 \\u 转义(启动器不会这么写)", b'{"model_dir": "D:\\\\hz-ci\\\\\\u4e2d\\u6587"}'),
                      ("值里有控制字符", b'{"model_dir": "D:\\\\hz-ci\\\\a\\nb"}')]:
    reset_world()
    install()
    write(os.path.join(DEF, "location.json"), content)
    write(os.path.join(ROOT, "中文", Q8), b"victim")
    uninstall(delete_data=True)
    check(f"location.json {name}:只删默认目录,不报错", not exists(DEF) and exists(ROOT, "中文", Q8))

# 共享路径(UNC)的模型目录:普通路径一样处理,只删我们的文件
unc_dir = "\\\\localhost\\D$\\hz-ci\\unc models"
reset_world()
if os.path.isdir("\\\\localhost\\D$"):
    install()
    write(os.path.join(unc_dir, Q8), b"GGUF")
    write(os.path.join(unc_dir, "other.txt"), b"keep")
    write(os.path.join(DEF, "location.json"), json.dumps({"model_dir": unc_dir}))
    uninstall(delete_data=True)
    check("UNC 模型目录:我们的模型删了、别的文件还在", not os.path.exists(os.path.join(unc_dir, Q8)) and os.path.exists(os.path.join(unc_dir, "other.txt")))
else:
    print("  (这台机器没有 \\\\localhost\\D$ 管理共享,跳过 UNC)")

print("7. 外部模型服务开着(0.3.4):改位置不拉起自带引擎,API key 跟着运行目录走;卸载删掉 key")
reset_world()
install()
EXT_PORT = 47831
fake = os.path.join(os.path.dirname(SETUP), "fakellama.exe")  # 当"外部服务"(和装进去的 llama-server.exe 不同名)
ext = subprocess.Popen([fake, "--port", str(EXT_PORT)], env=dict(os.environ, LLAMA_API_KEY="sk-win"),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
ext_run = ROOT + "\\外部 模式\\运行 目录"
try:
    write(os.path.join(DEF, "models", Q8), b"GGUF")
    write(os.path.join(DEF, "settings.json"), json.dumps({"tier": "q8"}))
    app = start()
    check("先用自带(假)引擎就绪", wait_phase("ready"))
    code, r = api("/app/external", {"enabled": True, "url": f"127.0.0.1:{EXT_PORT}", "model": "humanizer-12b-q8_0", "api_key": "sk-win"})
    st = api("/app/status")[1]
    check(f"开外部服务(HTTP {code}):就绪、backend=external、自带引擎进程没了",
          code == 200 and st["phase"] == "ready" and st["engine"]["backend"] == "external" and not procs("llama-server"))
    check("API key 在默认运行目录里", exists(DEF, "external-api-key"))
    code, r = api("/app/paths", {"data_dir": ext_run, "model_dir": "", "move": True})
    time.sleep(1)
    st = api("/app/status")[1]
    check(f"外部模式下改运行目录、搬模型(HTTP {code} {r.get('moved')})", code == 200 and r.get("moved") == 1 and exists(ext_run, "models", Q8))
    check("改完还是外部服务就绪,自带引擎没被拉起来",
          st["phase"] == "ready" and st["engine"]["backend"] == "external" and st["external"]["has_key"] and not procs("llama-server"))
    check("API key 跟到新运行目录,默认目录里没了", exists(ext_run, "external-api-key") and not exists(DEF, "external-api-key"))
    check("外部服务设置写进了新运行目录(不含 key)",
          f"127.0.0.1:{EXT_PORT}" in open(os.path.join(ext_run, "settings.json"), encoding="utf-8").read()
          and "sk-win" not in open(os.path.join(ext_run, "settings.json"), encoding="utf-8").read())
    code, r = api("/app/setup", {"tier": "q8"})
    check(f"外部模式下换档仍然被拒(HTTP {code})", code == 409)
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/api/v1/chat/completions", method="POST", headers={"X-Humanizer": "1", "Content-Type": "application/json"},
                                 data=json.dumps({"model": "humanizer-12b-q8_0", "messages": [{"role": "user", "content": "A short draft. It has two sentences."}],
                                                  "temperature": 1.0, "top_p": 0.95, "max_tokens": 256, "stream": True}).encode())
    with urllib.request.urlopen(req, timeout=60) as resp:
        sse = resp.read().decode("utf-8", "replace")
    check("改位置后改写经启动器反代到外部服务(带 key)", "chat.completion.chunk" in sse and "[DONE]" in sse)
    quit_app(app)
    app = start()
    check("重启后按 location.json 找到新运行目录,直接外部模式就绪", wait_phase("ready"))
    st = api("/app/status")[1]
    check("重启后 key 读得到、自带引擎没起", same(st["data_dir"], ext_run) and st["engine"]["backend"] == "external"
          and st["external"]["has_key"] and not procs("llama-server"))
    quit_app(app)
    uninstall(delete_data=True)
    check("卸载:默认目录删了", not exists(DEF))
    check("卸载:新运行目录里的 API key 删了,只有我们文件的运行目录整个没了", not exists(ext_run, "external-api-key") and not exists(ext_run))
    check("卸载没去动外部服务(它还在跑)", ext.poll() is None)
finally:
    ext.kill()

shutil.rmtree(ROOT, ignore_errors=True)
print()
print(f"通过 {passed} 项,失败 {len(failed)} 项")
for f in failed:
    print("  ✗", f)
sys.exit(1 if failed else 0)
