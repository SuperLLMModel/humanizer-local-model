; Humanizer Windows 安装包(Inno Setup 6)。
; CI 里这样编译(在 app/ 目录下):
;   iscc /DAppVersion=0.3.2 packaging\windows\humanizer.iss
; 输入:dist\win\ 下已经放好 Humanizer.exe 和 engine\{cuda,vulkan,cpu}\
; 输出:dist\Humanizer-<版本>-windows-x64-setup.exe
;
; 按用户安装(不需要管理员权限),装到 %LOCALAPPDATA%\Programs\Humanizer;
; 模型和日志默认在 %LOCALAPPDATA%\Humanizer(启动器管),可以在网页上的「存储位置」里改。
; 卸载时问要不要连数据一起删:默认目录整个删;location.json 里记的、用户自己选的目录只删 Humanizer 自己的文件。
; 静默卸载默认不删数据;设环境变量 HUMANIZER_UNINSTALL_DELETE_DATA=1 则不问直接删(脚本化卸载、CI 用)。

#ifndef AppVersion
  #define AppVersion "0.0.0-dev"
#endif

[Setup]
AppId={{5B7D3E4A-9C2F-4E61-A8D3-2F6B1C0E9A47}
AppName=Humanizer
AppVersion={#AppVersion}
AppVerName=Humanizer {#AppVersion}
AppPublisher=humanizer
AppPublisherURL=https://github.com/sgaofen/humanizer
DefaultDirName={localappdata}\Programs\Humanizer
DisableDirPage=auto
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\..\dist
OutputBaseFilename=Humanizer-{#AppVersion}-windows-x64-setup
SetupIconFile=..\icon\icon.ico
UninstallDisplayIcon={app}\Humanizer.exe
UninstallDisplayName=Humanizer
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; 升级/卸载时如果 Humanizer 正在跑,用 Restart Manager 关掉(引擎挂在它的 Job 上会一起退出)
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"
; 简体中文语言文件不是 Inno Setup 自带的官方翻译,CI 会先下载到 compiler 目录;没有就只出英文界面
#if FileExists(AddBackslash(CompilerPath) + "Languages\ChineseSimplified.isl")
Name: "zh"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
#endif

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\..\dist\win\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Humanizer"; Filename: "{app}\Humanizer.exe"
Name: "{autodesktop}\Humanizer"; Filename: "{app}\Humanizer.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Humanizer.exe"; Description: "{cm:LaunchProgram,Humanizer}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
; 只关 Humanizer.exe;不要按名字杀 llama-server.exe(可能误杀用户自己跑的)
Filename: "{sys}\taskkill.exe"; Parameters: "/f /im Humanizer.exe"; Flags: runhidden; RunOnceId: "StopHumanizer"

[CustomMessages]
en.AskDeleteModels=Also delete the downloaded models and logs?%n%n%1%n%nThe model files are large (several GB). Choose No to keep them for a later reinstall.
zh.AskDeleteModels=要不要把已下载的模型和日志也一起删掉？%n%n%1%n%n模型文件很大（几个 GB）。选「否」可以留着，下次重装直接用。
en.OnlyOurFiles=(only Humanizer's own files; anything else in this folder stays)
zh.OnlyOurFiles=（只删 Humanizer 自己的文件，这个文件夹里别的东西不动）

[Code]
// 卸载时清理数据。安全规则:
//   - 默认目录 %LOCALAPPDATA%\Humanizer 是我们自己的,整个删(DelTree 不会钻进符号链接/目录联接里删);
//   - location.json 里记的运行目录、模型目录是用户自己选的,可能和别的文件放在一起,
//     只按名字删 Humanizer 写过的文件,再删掉已经空了的文件夹;
//   - location.json 读不懂、路径不是普通的绝对路径、指到盘符根/用户目录/系统目录,一律当没设过。

// 从 location.json 的内容里取一个字符串字段。启动器写的是 UTF-8、只有 \\ 和 \" 两种转义
// (见 paths.go 的 locationJSON),这里只认这种写法;别的转义、控制字符一律返回空串(当没设过)。
function JsonString(const S, Key: String): String;
var
  I, N: Integer;
  C: Char;
  V: String;
begin
  Result := '';
  I := Pos('"' + Key + '"', S);
  if I = 0 then Exit;
  I := I + Length(Key) + 2;
  N := Length(S);
  while (I <= N) and ((S[I] = ' ') or (S[I] = ':') or (S[I] = #9) or (S[I] = #10) or (S[I] = #13)) do
    I := I + 1;
  if (I > N) or (S[I] <> '"') then Exit;
  I := I + 1;
  V := '';
  while I <= N do
  begin
    C := S[I];
    if C = '"' then
    begin
      Result := V;
      Exit;
    end;
    if Ord(C) < 32 then Exit;
    if C <> '\' then
    begin
      V := V + C;
      I := I + 1;
    end
    else
    begin
      if I = N then Exit;
      C := S[I + 1];
      if (C <> '"') and (C <> '\') then Exit;
      V := V + C;
      I := I + 2;
    end;
  end;
end;

// 是不是一个"普通"的绝对路径:X:\a\b 或 \\server\share\a;不能是盘符根、共享根,
// 不能带 /、.、..(手改坏或者故意篡改的 location.json)。
function IsPlainAbsDir(const S: String): Boolean;
var
  I, Slashes: Integer;
  T: String;
begin
  Result := False;
  T := S + '\';
  if (Length(S) < 4) or (Pos('/', S) > 0) or (Pos('\..\', T) > 0) or (Pos('\.\', T) > 0) or (Pos('\\', Copy(T, 3, Length(T))) > 0) then
    Exit;
  if (S[2] = ':') and (S[3] = '\') then
    Result := ((S[1] >= 'A') and (S[1] <= 'Z')) or ((S[1] >= 'a') and (S[1] <= 'z'))
  else if (Copy(S, 1, 2) = '\\') and (S[3] <> '?') and (S[3] <> '.') then
  begin
    Slashes := 0;
    for I := 1 to Length(S) do
      if S[I] = '\' then Slashes := Slashes + 1;
    Result := Slashes >= 4; // \\server\share\folder
  end;
end;

function SameDir(const A, B: String): Boolean;
begin
  Result := CompareText(RemoveBackslash(A), RemoveBackslash(B)) = 0;
end;

// A 在 B 里面(或就是 B)
function IsInside(const A, B: String): Boolean;
begin
  Result := Pos(Lowercase(AddBackslash(B)), Lowercase(AddBackslash(A))) = 1;
end;

// 用户目录、桌面、文档、系统目录这类大目录:运行目录不可能是它们(启动器只用空文件夹或自己建的子文件夹),
// location.json 指到这里只能是被改坏了,一律不碰。
function IsBigDir(const D: String): Boolean;
begin
  Result := SameDir(D, GetEnv('USERPROFILE')) or SameDir(D, ExpandConstant('{userdocs}')) or
    SameDir(D, ExpandConstant('{userdesktop}')) or SameDir(D, ExpandConstant('{localappdata}')) or
    SameDir(D, ExpandConstant('{userappdata}')) or SameDir(D, ExpandConstant('{win}')) or
    SameDir(D, ExpandConstant('{sys}')) or SameDir(D, ExpandConstant('{commonpf}')) or
    SameDir(D, ExpandConstant('{commonpf32}')) or SameDir(D, GetEnv('USERPROFILE') + '\Downloads') or
    SameDir(D, ExpandConstant('{localappdata}\Programs'));
end;

// Dir 里名字以 Prefix 开头(不分大小写)、又含 Part 的条目删掉;WithDirs 时同名的子目录也删。
// 名字再核一遍,不只靠通配符(Windows 的通配符还会去匹配 8.3 短文件名)。
procedure DeleteNamed(const Dir, Prefix, Part: String; WithDirs: Boolean);
var
  F: TFindRec;
  N: String;
begin
  if FindFirst(AddBackslash(Dir) + Prefix + '*', F) then
  try
    repeat
      N := Lowercase(F.Name);
      if (Pos(Lowercase(Prefix), N) = 1) and ((Part = '') or (Pos(Lowercase(Part), N) > 0)) then
      begin
        if (F.Attributes and FILE_ATTRIBUTE_DIRECTORY) = 0 then
          DeleteFile(AddBackslash(Dir) + F.Name)
        else if WithDirs then
          DelTree(AddBackslash(Dir) + F.Name, True, True, True);
      end;
    until not FindNext(F);
  finally
    FindClose(F);
  end;
end;

// 模型目录:只删档位文件 humanizer-*.gguf 及其半截/更新/备份文件(.part、.update、.bak…)。
procedure CleanModelDir(const Dir: String);
begin
  DeleteNamed(Dir, 'humanizer-', '.gguf', False);
  DeleteNamed(Dir + '\lite', 'humanizer-', '.gguf', False);
  RemoveDir(Dir + '\lite');
  RemoveDir(Dir);
end;

// 运行目录:只删启动器写的那几个文件和子目录里的东西,最后文件夹空了才删。
procedure DeleteWithTmp(const Dir, Name: String);
begin
  DeleteFile(Dir + '\' + Name);
  DeleteFile(Dir + '\' + Name + '.tmp'); // 原子写入时的临时文件
end;

procedure CleanRunDir(const Dir: String);
begin
  DeleteWithTmp(Dir, 'settings.json');
  DeleteWithTmp(Dir, 'config.json');
  DeleteWithTmp(Dir, 'instance.json');
  DeleteWithTmp(Dir, 'location.json');
  DeleteWithTmp(Dir, 'update-state.json');
  DeleteWithTmp(Dir, 'update-result.json');
  DeleteWithTmp(Dir, 'model-hashes.json');
  DeleteFile(Dir + '\.humanizer-write-test');
  DeleteNamed(Dir + '\logs', 'launcher.log', '', False);
  DeleteNamed(Dir + '\logs', 'llama-server.log', '', False);
  DeleteNamed(Dir + '\logs', 'update.log', '', False);
  DeleteNamed(Dir + '\logs', 'update-install.log', '', False);
  RemoveDir(Dir + '\logs');
  // 下载的新版安装包(Humanizer-<版本>-…)、更新助手(humanizer-updater…)和它的计划
  DeleteNamed(Dir + '\updates', 'humanizer-', '', True);
  DeleteFile(Dir + '\updates\apply-plan.json');
  RemoveDir(Dir + '\updates');
  CleanModelDir(Dir + '\models');
  RemoveDir(Dir);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Def, DataDir, ModelDir, List: String;
  Lines: TArrayOfString;
  Content: String;
  I: Integer;
  HasDef, HasData, HasModel: Boolean;
begin
  if CurUninstallStep <> usPostUninstall then Exit;

  Def := ExpandConstant('{localappdata}\Humanizer');
  // 先读 location.json:它就在默认目录里,删了就读不到了
  DataDir := '';
  ModelDir := '';
  if LoadStringsFromFile(Def + '\location.json', Lines) then // 按 UTF-8 读,中文路径不会乱
  begin
    Content := '';
    for I := 0 to GetArrayLength(Lines) - 1 do
      Content := Content + Lines[I] + #10;
    DataDir := RemoveBackslash(JsonString(Content, 'data_dir'));
    ModelDir := RemoveBackslash(JsonString(Content, 'model_dir'));
  end;

  Log(Format('location.json: data_dir=[%s] model_dir=[%s]', [DataDir, ModelDir]));
  HasDef := DirExists(Def);
  // 运行目录要有 logs\launcher.log(启动器每次启动都写)才认:防止 location.json 被改到一个不相干的目录
  HasData := IsPlainAbsDir(DataDir) and not IsBigDir(DataDir) and not (HasDef and IsInside(DataDir, Def)) and
    DirExists(DataDir) and FileExists(DataDir + '\logs\launcher.log');
  HasModel := IsPlainAbsDir(ModelDir) and not (HasDef and IsInside(ModelDir, Def)) and
    not (HasData and SameDir(ModelDir, DataDir + '\models')) and DirExists(ModelDir);
  Log(Format('要清理:默认目录=%d 运行目录=%d 模型目录=%d', [Ord(HasDef), Ord(HasData), Ord(HasModel)]));
  if not (HasDef or HasData or HasModel) then Exit;

  if GetEnv('HUMANIZER_UNINSTALL_DELETE_DATA') <> '1' then
  begin
    if UninstallSilent then Exit;
    List := '';
    if HasDef then List := List + Def + #13#10;
    if HasData then List := List + DataDir + ' ' + CustomMessage('OnlyOurFiles') + #13#10;
    if HasModel then List := List + ModelDir + ' ' + CustomMessage('OnlyOurFiles') + #13#10;
    if MsgBox(FmtMessage(CustomMessage('AskDeleteModels'), [Trim(List)]), mbConfirmation, MB_YESNO or MB_DEFBUTTON2) <> IDYES then
      Exit;
  end;

  if HasDef then DelTree(Def, True, True, True);
  if HasData then CleanRunDir(DataDir);
  if HasModel then CleanModelDir(ModelDir);
end;
