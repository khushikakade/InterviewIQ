document.addEventListener('DOMContentLoaded', () => {
  const resumeForm = document.getElementById('resumeUploadForm');
  const dropzone = document.getElementById('resumeUploadDropzone');
  const fileInput = document.getElementById('resumeFileInput');
  const jdForm = document.getElementById('jdMatchForm');

  // Drag and Drop Effects
  if (dropzone && fileInput) {
    ['dragenter', 'dragover'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
      }, false);
    });

    dropzone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files.length > 0) {
        fileInput.files = files;
        if (typeof showToast === 'function') {
          showToast(`Selected file: ${files[0].name}`);
        }
      }
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files.length > 0) {
        if (typeof showToast === 'function') {
          showToast(`Selected file: ${fileInput.files[0].name}`);
        }
      }
    });
  }

  // Handle Resume Upload Submit
  if (resumeForm) {
    resumeForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const btn = document.getElementById('uploadBtn');
      btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-2"></i> Parsing & Analyzing...';
      btn.disabled = true;

      const formData = new FormData(resumeForm);
      try {
        const res = await fetch('/api/resume/upload', {
          method: 'POST',
          body: formData
        });
        const result = await res.json();
        if (result.success) {
          if (typeof showToast === 'function') {
            showToast('Resume parsed & analyzed successfully!');
          }
          renderParsedResume(result.resume);
        } else {
          showToast(result.message || 'Error parsing resume', true);
        }
      } catch (err) {
        showToast('Network error during file upload', true);
      } finally {
        btn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles me-2"></i> Parse & Analyze Resume';
        btn.disabled = false;
      }
    });
  }

  // Handle Job Description Matching Form Submit
  if (jdForm) {
    jdForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const btn = document.getElementById('jdMatchBtn');
      const title = document.getElementById('jdTitleInput').value.trim();
      const jdText = document.getElementById('jdTextInput').value.trim();

      if (!jdText) {
        showToast('Please paste a Job Description first.', true);
        return;
      }

      btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-2"></i> Calculating Match...';
      btn.disabled = true;

      try {
        const res = await fetch('/api/jd/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            title: title || 'Software Developer',
            jd_text: jdText
          })
        });
        const result = await res.json();
        if (result.success && result.job_description) {
          showToast('Job Description matching completed!');
          renderJDMatchResult(result.job_description);
        } else {
          showToast(result.message || 'Error matching Job Description', true);
        }
      } catch (err) {
        showToast('Network error during JD match analysis', true);
      } finally {
        btn.innerHTML = '<i class="fa-solid fa-chart-line text-cyan me-2"></i> Calculate Skill Match Score';
        btn.disabled = false;
      }
    });
  }
});

function renderParsedResume(data) {
  const noPlaceholder = document.getElementById('noResumePlaceholder');
  const container = document.getElementById('resumeParsedContainer');
  if (noPlaceholder) noPlaceholder.classList.add('d-none');
  if (container) container.classList.remove('d-none');

  const parsed = (data && data.parsed_data) ? data.parsed_data : {};
  const filename = (data && data.filename) ? data.filename : 'Uploaded Resume';

  const nameEl = document.getElementById('parsedName');
  if (nameEl) nameEl.innerText = parsed.name || 'Candidate';

  const filenameEl = document.getElementById('parsedFilename');
  if (filenameEl) filenameEl.innerHTML = `<i class="fa-solid fa-file-code me-1 mt-1"></i> ${filename}`;

  const qualityScoreEl = document.getElementById('parsedQualityScore');
  if (qualityScoreEl) qualityScoreEl.innerText = `${parsed.quality_score || 85}%`;

  const wordCountEl = document.getElementById('parsedWordCount');
  if (wordCountEl) wordCountEl.innerText = parsed.word_count || 0;

  const skillsCountEl = document.getElementById('parsedSkillsCount');
  if (skillsCountEl) skillsCountEl.innerText = (parsed.skills || []).length;

  const projectsCountEl = document.getElementById('parsedProjectsCount');
  if (projectsCountEl) projectsCountEl.innerText = (parsed.projects || []).length;

  const skillsDiv = document.getElementById('parsedSkills');
  if (skillsDiv) {
    if (parsed.skills && parsed.skills.length > 0) {
      skillsDiv.innerHTML = parsed.skills.map(s => `<span class="skill-tag">${escapeHtml(s)}</span>`).join('');
    } else {
      skillsDiv.innerHTML = '<span class="text-muted small">No specific skills detected.</span>';
    }
  }

  const techDiv = document.getElementById('parsedTechnologies');
  if (techDiv) {
    if (parsed.technologies && parsed.technologies.length > 0) {
      techDiv.innerHTML = parsed.technologies.map(t => `<span class="stitch-badge bg-dark border border-subtle text-main me-1 mb-1">${escapeHtml(t)}</span>`).join('');
    } else {
      techDiv.innerHTML = '<span class="text-muted small">General Software Engineering</span>';
    }
  }

  const eduUl = document.getElementById('parsedEducation');
  if (eduUl) {
    if (parsed.education && parsed.education.length > 0) {
      eduUl.innerHTML = parsed.education.map(e => `<li class="mb-1">${escapeHtml(e)}</li>`).join('');
    } else {
      eduUl.innerHTML = '<li>Higher Education in CS/Engineering</li>';
    }
  }

  const expUl = document.getElementById('parsedExperience');
  if (expUl) {
    if (parsed.experience && parsed.experience.length > 0) {
      expUl.innerHTML = parsed.experience.map(ex => `<li class="mb-1">${escapeHtml(ex)}</li>`).join('');
    } else {
      expUl.innerHTML = '<li>Engineering Candidate</li>';
    }
  }

  const projUl = document.getElementById('parsedProjects');
  if (projUl) {
    if (parsed.projects && parsed.projects.length > 0) {
      projUl.innerHTML = parsed.projects.map(p => `<li class="mb-1">${escapeHtml(p)}</li>`).join('');
    } else {
      projUl.innerHTML = '<li>Technical Project Showcase</li>';
    }
  }

  const certUl = document.getElementById('parsedCertifications');
  if (certUl) {
    if (parsed.certifications && parsed.certifications.length > 0) {
      certUl.innerHTML = parsed.certifications.map(c => `<li class="mb-1">${escapeHtml(c)}</li>`).join('');
    } else {
      certUl.innerHTML = '<li class="text-secondary italic">No certifications detected.</li>';
    }
  }
}

function renderJDMatchResult(jdData) {
  const card = document.getElementById('jdMatchResultCard');
  if (!card) return;

  card.classList.remove('d-none');
  
  const scoreBadge = document.getElementById('jdMatchScoreBadge');
  const score = jdData.match_score || 0;
  if (scoreBadge) {
    scoreBadge.innerText = `${score}% Skill Match`;
    if (score >= 75) {
      scoreBadge.className = 'stitch-badge badge-ready fs-5';
    } else if (score >= 55) {
      scoreBadge.className = 'stitch-badge badge-moderate fs-5';
    } else {
      scoreBadge.className = 'stitch-badge badge-needs-practice fs-5';
    }
  }

  const details = jdData.match_details || {};
  const matchedList = details.matched_skills || [];
  const missingList = details.missing_skills || [];

  const matchedDiv = document.getElementById('jdMatchedSkillsList');
  if (matchedDiv) {
    if (matchedList.length > 0) {
      matchedDiv.innerHTML = matchedList.map(s => `<span class="skill-tag">${escapeHtml(s)}</span>`).join('');
    } else {
      matchedDiv.innerHTML = '<span class="text-muted small">No exact skill matches detected.</span>';
    }
  }

  const missingDiv = document.getElementById('jdMissingSkillsList');
  if (missingDiv) {
    if (missingList.length > 0) {
      missingDiv.innerHTML = missingList.map(s => `<span class="skill-tag-missing">${escapeHtml(s)}</span>`).join('');
    } else {
      missingDiv.innerHTML = '<span class="text-emerald small"><i class="fa-solid fa-check me-1"></i> All core job skills matched!</span>';
    }
  }

  card.scrollIntoView({ behavior: 'smooth' });
}

function escapeHtml(text) {
  if (!text) return '';
  const div = document.createElement('div');
  div.innerText = text;
  return div.innerHTML;
}
