// Common Toast Notification Helper
function showToast(message, isError = false) {
  const toastEl = document.getElementById('liveToast');
  const toastMsg = document.getElementById('toastMessage');

  if (toastEl && toastMsg) {
    toastMsg.innerText = message;
    if (isError) {
      toastEl.classList.remove('text-bg-dark');
      toastEl.classList.add('text-bg-danger');
    } else {
      toastEl.classList.remove('text-bg-danger');
      toastEl.classList.add('text-bg-dark');
    }
    const toast = new bootstrap.Toast(toastEl);
    toast.show();
  } else {
    alert(message);
  }
}
