//go:build !windows

package launcher

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os/exec"
	"runtime"
	"strings"
)

// pickDir 弹系统的「选文件夹」窗口。用户取消时返回空字符串;没有可用工具时报错(网页提示直接粘路径)。
// 只在本机弹窗,不联网;标题和起始目录通过参数/env 传,不拼进脚本字符串。
func pickDir(ctx context.Context, from, title string) (string, error) {
	if from == "" {
		from = "."
	}
	if runtime.GOOS == "darwin" {
		return osascriptPick(ctx, from, title)
	}
	for _, c := range [][]string{
		{"zenity", "--file-selection", "--directory", "--title=" + title, "--filename=" + strings.TrimSuffix(from, "/") + "/"},
		{"kdialog", "--title=" + title, "--getexistingdirectory", from},
	} {
		if _, lookErr := exec.LookPath(c[0]); lookErr != nil {
			continue
		}
		out, err := exec.CommandContext(ctx, c[0], c[1:]...).Output()
		if ctx.Err() != nil {
			return "", ctx.Err()
		}
		var exit *exec.ExitError
		if err != nil && !errors.As(err, &exit) {
			continue // 起不来,换下一个
		}
		// 用户取消:退出码非 0、没有输出 → 空路径。别再弹下一个工具的窗口
		return absDir(strings.TrimSpace(string(out))), nil
	}
	return "", errors.New("没有可用的文件夹选择器(zenity / kdialog 都没装)")
}

// osascriptPick macOS 的原生选择器(可以 ⇧⌘G 直接输入路径)。用户点取消返回空路径。
func osascriptPick(ctx context.Context, from, title string) (string, error) {
	script := `on run argv
	set p to (choose folder with prompt (item 1 of argv) default location (POSIX file (item 2 of argv)))
	return POSIX path of p
end run`
	var stderr bytes.Buffer
	cmd := exec.CommandContext(ctx, "osascript", "-e", script, title, from)
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	switch {
	case ctx.Err() != nil:
		return "", ctx.Err()
	case err != nil && strings.Contains(stderr.String(), "-128"): // User canceled.
		return "", nil
	case err != nil:
		return "", fmt.Errorf("%v %s", err, strings.TrimSpace(stderr.String()))
	}
	return absDir(strings.TrimSpace(string(out))), nil
}
