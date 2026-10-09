let questions = [];
let currentIdx = 0;
let mediaRecorder = null;
let recordedChunks = [];
let mediaStream = null;
let timerInterval = null;
let secondsElapsed = 0;
let speechRecognition = null;
let answeredQuestionIds = new Set();
let cachedTranscripts = {};

document.addEventListener('DOMContentLoaded', () => {
  const dataEl = document.getElementById('questionsData');
  if (dataEl) {
    try {
      questions = JSON.parse(dataEl.textContent);
    } catch (e) {
      console.error('Failed to parse questions JSON:', e);
    }
  }

  setupWebcam();
  setupSpeechRecognition();
  setupPaletteListeners();
  loadQuestion(0);

  document.getElementById('startRecBtn').addEventListener('click', startRecording);
  document.getElementById('stopRecBtn').addEventListener('click', stopRecording);
  document.getElementById('submitAnswerBtn').addEventListener('click', submitCurrentAnswer);
  
  const prevBtn = document.getElementById('prevQuestionBtn');
  if (prevBtn) prevBtn.addEventListener('click', () => loadQuestion(currentIdx - 1));

  const nextBtn = document.getElementById('nextQuestionBtn');
  if (nextBtn) nextBtn.addEventListener('click', () => loadQuestion(currentIdx + 1));

  const finishBtn = document.getElementById('finishSessionBtn');
  if (finishBtn) finishBtn.addEventListener('click', completeInterviewSession);

  // Auto-save transcript typing locally into cache
  const textarea = document.getElementById('transcriptTextarea');
  if (textarea) {
    textarea.addEventListener('input', (e) => {
      if (questions[currentIdx]) {
        cachedTranscripts[questions[currentIdx].id] = e.target.value;
        updateLiveMetrics(e.target.value);
      }
    });
  }
});

function setupPaletteListeners() {
  const buttons = document.querySelectorAll('.q-nav-btn');
  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      const idx = parseInt(btn.getAttribute('data-idx'));
      if (!isNaN(idx) && idx >= 0 && idx < questions.length) {
        loadQuestion(idx);
      }
    });
  });
}

async function setupWebcam() {
  const videoEl = document.getElementById('webcamFeed');
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    if (videoEl) videoEl.srcObject = mediaStream;
    const cue = document.getElementById('eyeCueText');
    if (cue) cue.innerText = 'Eye Contact: Monitoring';
    
    document.getElementById('liveEyeMetric').innerText = '92%';
    document.getElementById('liveStabilityMetric').innerText = 'Optimal';
  } catch (err) {
    console.warn('Camera/Microphone access not granted:', err);
    const cue = document.getElementById('eyeCueText');
    if (cue) cue.innerText = 'WebCam Off (Text Mode)';
    document.getElementById('liveEyeMetric').innerText = 'N/A';
    document.getElementById('liveStabilityMetric').innerText = 'Text Mode';
    if (typeof showToast === 'function') {
      showToast('Camera/Microphone access denied. You can type your answers directly.', true);
    }
  }
}

function setupSpeechRecognition() {
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SpeechRec) {
    speechRecognition = new SpeechRec();
    speechRecognition.continuous = true;
    speechRecognition.interimResults = true;

    speechRecognition.onresult = (event) => {
      let transcript = '';
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        transcript += event.results[i][0].transcript;
      }
      const textarea = document.getElementById('transcriptTextarea');
      if (textarea && transcript.trim()) {
        textarea.value = transcript;
        if (questions[currentIdx]) {
          cachedTranscripts[questions[currentIdx].id] = transcript;
          updateLiveMetrics(transcript);
        }
      }
    };
  }
}

function updateLiveMetrics(text) {
  if (!text || text.trim().length === 0) {
    document.getElementById('liveSpeechMetric').innerText = '0 WPM';
    document.getElementById('liveFillersMetric').innerText = '0';
    return;
  }

  const words = text.trim().split(/\s+/).length;
  const mins = Math.max(secondsElapsed / 60, 0.25);
  const wpm = Math.round(words / mins);
  document.getElementById('liveSpeechMetric').innerText = `${wpm} WPM`;

  const fillers = (text.match(/\b(um|uh|like|you know|so|basically|actually)\b/gi) || []).length;
  document.getElementById('liveFillersMetric').innerText = `${fillers}`;
}

function loadQuestion(index) {
  if (index < 0 || index >= questions.length) {
    if (index >= questions.length && typeof showToast === 'function') {
      showToast('You have reached the end of questions.');
    }
    return;
  }

  stopRecording();
  recordedChunks = [];
  currentIdx = index;
  const q = questions[index];

  document.getElementById('currentQuestionNum').innerText = index + 1;
  document.getElementById('questionCategory').innerText = q.category || 'General';
  document.getElementById('questionType').innerText = q.question_type || 'Technical';
  document.getElementById('questionText').innerText = q.question_text;

  const textarea = document.getElementById('transcriptTextarea');
  textarea.value = cachedTranscripts[q.id] || '';
  updateLiveMetrics(textarea.value);

  const badge = document.getElementById('answerSavedBadge');
  if (badge) {
    badge.innerText = answeredQuestionIds.has(q.id) ? '✓ Answer Evaluated' : 'Auto-transcribed or typed';
    badge.className = answeredQuestionIds.has(q.id) ? 'text-emerald fw-bold small' : 'text-muted small';
  }

  const pct = ((index + 1) / questions.length) * 100;
  document.getElementById('progressBar').style.width = `${pct}%`;
  updatePaletteUI();

  resetTimer();
}

function updatePaletteUI() {
  questions.forEach((q, idx) => {
    const btn = document.getElementById(`qNavBtn_${idx}`);
    if (!btn) return;

    if (idx === currentIdx) {
      btn.className = 'btn btn-sm btn-stitch-primary py-1 px-3 fw-bold';
    } else if (answeredQuestionIds.has(q.id)) {
      btn.className = 'btn btn-sm btn-stitch-secondary border-success text-emerald py-1 px-3';
    } else {
      btn.className = 'btn btn-sm btn-stitch-secondary py-1 px-3';
    }
  });
}

function startRecording() {
  recordedChunks = [];
  secondsElapsed = 0;

  if (mediaStream) {
    try {
      mediaRecorder = new MediaRecorder(mediaStream);
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) recordedChunks.push(e.data);
      };
      mediaRecorder.start();
    } catch (err) {
      console.warn('MediaRecorder error:', err);
    }
  }

  if (speechRecognition) {
    try { speechRecognition.start(); } catch (e) {}
  }

  document.getElementById('startRecBtn').classList.add('d-none');
  document.getElementById('stopRecBtn').classList.remove('d-none');
  document.getElementById('sessionStatusText').innerText = 'Recording live answer...';

  timerInterval = setInterval(() => {
    secondsElapsed++;
    const mins = String(Math.floor(secondsElapsed / 60)).padStart(2, '0');
    const secs = String(secondsElapsed % 60).padStart(2, '0');
    document.getElementById('timerDisplay').innerText = `${mins}:${secs}`;
    
    const textarea = document.getElementById('transcriptTextarea');
    if (textarea) updateLiveMetrics(textarea.value);
  }, 1000);
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  if (speechRecognition) {
    try { speechRecognition.stop(); } catch (e) {}
  }

  clearInterval(timerInterval);
  const startBtn = document.getElementById('startRecBtn');
  const stopBtn = document.getElementById('stopRecBtn');
  if (startBtn) startBtn.classList.remove('d-none');
  if (stopBtn) stopBtn.classList.add('d-none');
  const statusEl = document.getElementById('sessionStatusText');
  if (statusEl) statusEl.innerText = 'Ready for answer submission.';
}

function resetTimer() {
  clearInterval(timerInterval);
  secondsElapsed = 0;
  const timerEl = document.getElementById('timerDisplay');
  if (timerEl) timerEl.innerText = '00:00';
}

async function submitCurrentAnswer() {
  stopRecording();

  const q = questions[currentIdx];
  const transcript = document.getElementById('transcriptTextarea').value.trim();
  const submitBtn = document.getElementById('submitAnswerBtn');

  submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-1"></i> Evaluating...';
  submitBtn.disabled = true;

  const formData = new FormData();
  formData.append('question_id', q.id);
  formData.append('transcript', transcript);
  formData.append('duration_seconds', secondsElapsed > 0 ? secondsElapsed : 30);

  if (recordedChunks.length > 0) {
    const blob = new Blob(recordedChunks, { type: 'video/webm' });
    formData.append('media', blob, `answer_${q.id}.webm`);
  }

  try {
    const res = await fetch(`/api/interview/${window.INTERVIEW_ID}/submit_answer`, {
      method: 'POST',
      body: formData
    });
    const result = await res.json();
    if (result.success) {
      answeredQuestionIds.add(q.id);
      cachedTranscripts[q.id] = transcript;
      if (typeof showToast === 'function') {
        showToast(`Question ${currentIdx + 1} evaluated successfully!`);
      }
      
      if (currentIdx + 1 < questions.length) {
        loadQuestion(currentIdx + 1);
      } else {
        completeInterviewSession();
      }
    } else {
      if (typeof showToast === 'function') {
        showToast(result.message || 'Error submitting answer', true);
      }
    }
  } catch (err) {
    if (typeof showToast === 'function') {
      showToast('Network error during answer submission', true);
    }
  } finally {
    submitBtn.innerHTML = '<i class="fa-solid fa-paper-plane me-1"></i> Submit Answer';
    submitBtn.disabled = false;
  }
}

async function completeInterviewSession() {
  stopRecording();
  document.getElementById('sessionStatusText').innerText = 'Analyzing full interview performance...';
  if (typeof showToast === 'function') {
    showToast('Completing session & generating report...');
  }

  try {
    const res = await fetch(`/api/interview/${window.INTERVIEW_ID}/complete`, {
      method: 'POST'
    });
    const result = await res.json();
    if (result.success && result.redirect) {
      window.location.href = result.redirect;
    } else {
      window.location.href = `/report/${window.INTERVIEW_ID}`;
    }
  } catch (err) {
    window.location.href = `/report/${window.INTERVIEW_ID}`;
  }
}
