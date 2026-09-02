const API_BASE = 'https://deepfake-detector-ig87.onrender.com';

document.addEventListener('DOMContentLoaded', () => {
    const signInButton = document.querySelector('[data-auth-target]');

    if (signInButton) {
        signInButton.addEventListener('click', () => {
            window.location.href = signInButton.getAttribute('data-auth-target');
        });
    }

    // Intercept protected links and direct unauthenticated users to login
    document.querySelectorAll('a[href="/dashboard"], a[href="/history"]').forEach(link => {
        link.addEventListener('click', async (e) => {
            e.preventDefault();
            const targetHref = link.getAttribute('href');
            try {
                const res = await fetch(`${API_BASE}/auth/me`, { credentials: 'include' });
                if (res.ok) {
                    const data = await res.json();
                    if (data && data.authenticated) {
                        window.location.href = targetHref;
                        return;
                    }
                }
            } catch (err) {
                console.warn('Auth check on navigation error:', err);
            }
            window.location.href = `/login?returnTo=${encodeURIComponent(targetHref)}&reason=Please+sign+in+to+access+the+platform.`;
        });
    });
});
