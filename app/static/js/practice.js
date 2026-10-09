let practiceQuestions = [];
let currentPracticeIdx = 0;

document.addEventListener('DOMContentLoaded', () => {
  const dataEl = document.getElementById('practiceData');
  if (dataEl) {
    try {
      practiceQuestions = JSON.parse(dataEl.textContent);
    } catch (e) {
      console.error('Failed to parse practice questions JSON:', e);
    }
  }

  if (!practiceQuestions || practiceQuestions.length === 0) {
    practiceQuestions = [{
      weakness: 'Technical Depth',
      question_text: 'Explain how you optimize a slow SQL query with indexes, execution plans, and caching.'
    }];
  }

  loadPracticeQuestion(0);

  const nextBtn = document.getElementById('nextPracticeQBtn');
  if (nextBtn) {
    nextBtn.addEventListener('click', () => {
      currentPracticeIdx = (currentPracticeIdx + 1) % practiceQuestions.length;
      loadPracticeQuestion(currentPracticeIdx);
    });
  }

  const practiceForm = document.getElementById('practiceForm');
  if (practiceForm) {
    practiceForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const btn = document.getElementById('submitPracticeBtn');
      btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-1"></i> Re-Evaluating...';
      btn.disabled = true;

      const q = practiceQuestions[currentPracticeIdx];
      const answerText = document.getElementById('practiceAnswerInput').value.trim();

      try {
        const res = await fetch('/api/practice/submit', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            target_weakness: q.weakness,
            question_text: q.question_text,
            user_answer: answerText
          })
        });
        const result = await res.json();
        if (result.success) {
          if (typeof showToast === 'function') showToast('Practice response evaluated!');
          const box = document.getElementById('practiceResultBox');
          document.getElementById('practiceFeedbackText').innerText = result.session.feedback_text;
          box.classList.remove('d-none');
        } else {
          if (typeof showToast === 'function') showToast(result.message || 'Error evaluating practice response', true);
        }
      } catch (err) {
        if (typeof showToast === 'function') showToast('Network error during evaluation', true);
      } finally {
        btn.innerHTML = '<i class="fa-solid fa-check-double me-1"></i> Re-Evaluate Response';
        btn.disabled = false;
      }
    });
  }
});

function loadPracticeQuestion(index) {
  currentPracticeIdx = index;
  const q = practiceQuestions[index];

  const tag = document.getElementById('practiceWeaknessTag');
  if (tag) tag.innerText = `Target: ${q.weakness}`;

  const textEl = document.getElementById('practiceQuestionText');
  if (textEl) textEl.innerText = q.question_text;

  const inputEl = document.getElementById('practiceAnswerInput');
  if (inputEl) inputEl.value = '';

  const box = document.getElementById('practiceResultBox');
  if (box) box.classList.add('d-none');
}
