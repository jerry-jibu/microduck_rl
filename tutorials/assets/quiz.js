/* Microduck RL 课程共享测验组件。
   用法：
   <div class="quiz" data-answer="b">
     <span class="qno">测验 1</span>
     <p class="q">问题文本？</p>
     <div class="opts">
       <button data-k="a">选项甲</button>
       <button data-k="b">选项乙</button>
       <button data-k="c">选项丙</button>
     </div>
     <div class="explain" data-good="答对的解释" data-bad="答错的解释">
       <span class="ref">出处：<a href="...">…</a></span>
     </div>
   </div>
   选项按钮在加载时随机排序，避免位置提示。 */
(function () {
  function shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }
  window.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".quiz").forEach((quiz) => {
      const answer = quiz.dataset.answer;
      const opts = quiz.querySelector(".opts");
      const explain = quiz.querySelector(".explain");
      shuffle(Array.from(opts.children)).forEach((b) => opts.appendChild(b));
      opts.addEventListener("click", (e) => {
        const btn = e.target.closest("button");
        if (!btn || opts.dataset.locked) return;
        opts.dataset.locked = "1";
        const good = btn.dataset.k === answer;
        btn.classList.add(good ? "correct" : "wrong");
        if (!good) {
          Array.from(opts.children)
            .find((b) => b.dataset.k === answer)
            .classList.add("correct");
        }
        Array.from(opts.children).forEach((b) => (b.disabled = true));
        explain.classList.add(good ? "good" : "bad");
        explain.insertBefore(
          document.createTextNode(good ? "✓ 正确。" : "✗ 再想想。"),
          explain.firstChild
        );
      });
    });
  });
})();
