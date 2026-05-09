# 🤢 CPR Repository

## 💩 Repository 구조

```text
cpr_repo/
├── rppg_ws/
└── cpr_ws/
    └── src/
        ├── EAR/
        ├── motor/
        ├── Loadcell/
        ├── rppg_bridge_pkg/
        └── ui_pkg/
```

---

# 🤢 rppg_ws Usage

`rppg_ws`는 실시간 rPPG 및 UI 코드입니다.

GitHub에는 용량 문제로 아래 폴더가 포함 X

- dataset/
- log/
- runs/

위 폴더들은 팀 Notion에서 다운로드 후 `rppg_ws/` 내부에 위치

예시:

```text
rppg_ws/
├── dataset/
├── log/
├── runs/
```

---

# 🤢 cpr_ws Usage (ROS2 워크스페이스)

`cpr_ws`는 ROS2 workspace

## 💩 Build

```bash
# 해당 폴더에 들어가서
cd cpr_repo/cpr_ws
# 빌드 해야함
colcon build
```

## 💩 Source

```bash
# 해당 명령어를 사용하여 업데이트
source install/setup.bash
```

---

# 💩 예시

```bash
git clone https://github.com/REBOOT-GADGET21/cpr_repo.git

cd cpr_repo/cpr_ws
colcon build
source install/setup.bash
```
