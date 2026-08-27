document.addEventListener('DOMContentLoaded', () => {
    const signInButton = document.querySelector('[data-auth-target]');

    if (signInButton) {
        signInButton.addEventListener('click', () => {
            window.location.href = signInButton.getAttribute('data-auth-target');
        });
    }
});
