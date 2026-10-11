package launcher

import (
	"context"
	"os/exec"
	"strings"
	"testing"
	"time"
)

// 选择器窗口本身在 CI 里点不了,这里测外面那层:中文/空格路径原样进出、找不到 powershell 时报错而不是卡住。
func TestPickScriptRoundTripsUnicode(t *testing.T) {
	if _, err := exec.LookPath("powershell.exe"); err != nil {
		t.Skip("没有 powershell.exe")
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	from := `D:\模型 文件夹\R&D 'x' "y" $env:PATH ; calc`
	got, err := runPickScript(ctx, pickPrelude+`[Console]::Out.Write($env:HZ_PICK_START + '|' + $env:HZ_PICK_TITLE)`, from, "选择模型文件夹")
	if err != nil {
		t.Fatal(err)
	}
	// runPickScript 会把结果当路径收拾一遍(absDir),所以这里比较收拾前后都一样的部分
	if !strings.Contains(got, `模型 文件夹\R&D 'x' "y" $env:PATH ; calc|选择模型文件夹`) {
		t.Fatalf("路径/标题没原样传回来(被当成脚本执行或编码错了):%q", got)
	}
}

func TestPickScriptReportsFailure(t *testing.T) {
	if _, err := exec.LookPath("powershell.exe"); err != nil {
		t.Skip("没有 powershell.exe")
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	_, err := runPickScript(ctx, pickPrelude+`[Console]::Error.WriteLine('现代选择器: 坏了'); exit 1`, "", "")
	if err == nil || !strings.Contains(err.Error(), "现代选择器: 坏了") {
		t.Fatalf("失败原因应原样(UTF-8)带回来:%v", err)
	}
}

func TestPickScriptTimeout(t *testing.T) {
	if _, err := exec.LookPath("powershell.exe"); err != nil {
		t.Skip("没有 powershell.exe")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	start := time.Now()
	_, err := runPickScript(ctx, `Start-Sleep -Seconds 60`, "", "")
	if err == nil || time.Since(start) > 30*time.Second {
		t.Fatalf("超时应被打断并报错:%v(用时 %v)", err, time.Since(start))
	}
}

func TestPickScriptParses(t *testing.T) {
	if _, err := exec.LookPath("powershell.exe"); err != nil {
		t.Skip("没有 powershell.exe")
	}
	// 只解析不执行:脚本有语法错误的话,三个选择器一个都不会弹
	check := `$e = $null; [void][System.Management.Automation.Language.Parser]::ParseInput($env:HZ_PICK_START, [ref]$null, [ref]$e); if ($e.Count) { [Console]::Error.WriteLine($e[0].ToString()); exit 1 }; [Console]::Out.Write('ok')`
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, "powershell.exe", "-NoProfile", "-NonInteractive", "-Command", check)
	cmd.Env = append(cmd.Environ(), "HZ_PICK_START="+pickPrelude+pickScript)
	out, err := cmd.CombinedOutput()
	if err != nil || !strings.Contains(string(out), "ok") {
		t.Fatalf("选择器脚本解析失败:%v %s", err, out)
	}
}
