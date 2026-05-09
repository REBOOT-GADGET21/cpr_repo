# CPR Repository

## Repository Structure

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

# rppg_ws Usage

`rppg_ws`는 실시간 rPPG 및 UI 코드입니다.

GitHub에는 용량 문제로 아래 폴더가 포함되어 있지 않습니다.

- dataset/
- log/
- runs/

위 폴더들은 팀 Notion에서 다운로드 후 `rppg_ws/` 내부에 위치시켜야 합니다.

예시:

```text
rppg_ws/
├── dataset/
├── log/
├── runs/
```

---

# cpr_ws Usage (ROS2 Workspace)

`cpr_ws`는 ROS2 workspace입니다.

## Build

```bash
cd cpr_repo/cpr_ws
colcon build
```

## Source

```bash
source install/setup.bash
```

---

# Example

```bash
git clone https://github.com/REBOOT-GADGET21/cpr_repo.git

cd cpr_repo/cpr_ws
colcon build
source install/setup.bash
```
