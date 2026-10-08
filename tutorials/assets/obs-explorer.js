/* 观测探索器：把 61 维观测渲染成可点击的彩色分段网格。
   用法：
     <div class="obs-explorer"></div>
     <div class="obs-panel" id="obs-info">点击任意一格查看该维度的含义</div>
   分段定义与本仓库 microduck_brax_env.py 的 _get_obs 严格一致——
   改观测布局时必须同步改这里。 */
(function () {
  const SEGMENTS = [
    { name: "joint_pos", start: 0, end: 14, color: "#a2542a",
      src: "qpos[7:21] − STAND关节角",
      desc: "14 个关节角度，单位弧度，是相对标准站姿的偏移——0 = 摆好站姿。 hip/膝/踝/脖子的姿态全在这里。" },
    { name: "joint_vel", start: 14, end: 28, color: "#c77b45",
      src: "qvel[6:20]",
      desc: "14 个关节角速度（弧度/秒）。动作多剧烈，看它而不是看角度。" },
    { name: "base_ang_vel", start: 28, end: 31, color: "#d9a24a",
      src: "qvel[3:6]（机体系，已实测锁定）",
      desc: "躯干旋转快慢。注意帧语义：自由关节 qvel[0:3] 是世界系线速度，qvel[3:6] 是机体系角速度——与 IMU 陀螺仪读数同帧。" },
    { name: "proj_grav", start: 31, end: 34, color: "#7d9a5d",
      src: "四元数逆旋转 [0,0,−1]",
      desc: "内置倾斜计：世界重力投影到机身坐标。站直 = (0,0,−1)；前倾则 x 分量增大。摔倒判定（>−0.5 即约 60° 倾角）就用它。" },
    { name: "last_action", start: 34, end: 48, color: "#5d7d9a",
      src: "上一步 clip(±1) 后的动作",
      desc: "动作缓存。让网络知道自己刚发过什么指令，是动作平滑的现实基础。" },
    { name: "command", start: 48, end: 61, color: "#8a6aa0",
      src: "reset 时采样 vx∈[0,0.6]，其余恒零",
      desc: "13 维指令块：[vx, vy, wz] + 头部姿态(4) + 身体姿态(6)。当前任务只用前进速度 vx；其余 10 个槽零填充占位——这是全家族策略热插拔的接口契约，永不删槽。" },
  ];

  window.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".obs-explorer").forEach((grid) => {
      const panel = document.getElementById(grid.dataset.panel || "obs-info");
      const segOf = (i) => SEGMENTS.find((s) => i >= s.start && i < s.end);
      for (let i = 0; i < 61; i++) {
        const seg = segOf(i);
        const cell = document.createElement("div");
        cell.className = "cell";
        cell.style.background = seg.color;
        cell.textContent = i;
        cell.title = `${seg.name}[${i - seg.start}]`;
        cell.addEventListener("click", () => {
          grid.querySelectorAll(".cell.sel").forEach((c) => c.classList.remove("sel"));
          cell.classList.add("sel");
          panel.innerHTML =
            `<b class="seg">${seg.name}</b> <span class="idx">obs[${seg.start}:${seg.end}] · 当前 obs[${i}] · 来源 ${seg.src}</span><br>${seg.desc}`;
        });
        grid.appendChild(cell);
      }
      const legend = document.createElement("div");
      legend.className = "obs-legend";
      legend.innerHTML = SEGMENTS.map(
        (s) => `<span><i style="background:${s.color}"></i>${s.name}(${s.end - s.start})</span>`
      ).join("");
      grid.appendChild(legend);
    });
  });
})();
