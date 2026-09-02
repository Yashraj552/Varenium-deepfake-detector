const API_BASE = 'https://deepfake-detector-ig87.onrender.com';

function initializeFirebaseAuth() {
    const button = document.getElementById('google-signin-btn');
    if (!button) return;

    button.addEventListener('click', async () => {
        const message = document.getElementById('auth-message');
        button.disabled = true;
        message.textContent = 'CONNECTING TO GOOGLE...';

        try {
            if (!window.FIREBASE_CONFIG || window.FIREBASE_CONFIG.apiKey.startsWith('PASTE_')) {
                throw new Error('Firebase is not configured. Add your Firebase web app values to firebase-config.js.');
            }

            if (!firebase.apps.length) firebase.initializeApp(window.FIREBASE_CONFIG);
            await firebase.auth().setPersistence(firebase.auth.Auth.Persistence.SESSION);
            const provider = new firebase.auth.GoogleAuthProvider();
            provider.setCustomParameters({ prompt: 'select_account' });
            
            const result = await firebase.auth().signInWithPopup(provider);
            const idToken = await result.user.getIdToken();
            const response = await fetch(`${API_BASE}/auth/firebase`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ id_token: idToken })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Firebase sign-in failed.');
            window.location.href = getReturnPath();
        } catch (error) {
            console.error('Sign-in error:', error);
            if (error.code === 'auth/popup-closed-by-user' || error.code === 'auth/cancelled-popup-request') {
                message.textContent = 'Sign-in popup was closed before completing. Please try again.';
            } else if (error.code === 'auth/unauthorized-domain') {
                message.textContent = 'This domain is not authorized in Firebase Console yet. Please add it in Authorized Domains.';
            } else if (error.code === 'auth/popup-blocked') {
                message.textContent = 'Popup was blocked by your browser. Please enable popups for this site and try again.';
            } else {
                message.textContent = error.message || 'Authentication failed. Please try again.';
            }
            button.disabled = false;
        }
    });
}

function showLoginContext() {
    const params = new URLSearchParams(window.location.search);
    const reason = params.get('reason') || params.get('error');
    const message = document.getElementById('auth-message');
    if (reason && message) message.textContent = reason;
}

function getReturnPath() {
    const returnTo = new URLSearchParams(window.location.search).get('returnTo');
    if (returnTo && (returnTo === '/dashboard' || returnTo === 'dashboard.html' || returnTo === 'dashboard')) {
        return '/dashboard';
    }
    return '/dashboard';
}

async function submitAuthForm(event, endpoint) {
    event.preventDefault();
    const form = event.currentTarget;
    const message = document.getElementById('auth-message');
    const submitButton = form.querySelector('button[type="submit"]');
    const payload = Object.fromEntries(new FormData(form).entries());

    submitButton.disabled = true;
    message.textContent = 'PROCESSING...';
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify(payload)
        });
        const data = await response.json();

        if (response.status === 429) {
            throw new Error(data.error || 'Too many failed login attempts. Please try again in a few minutes.');
        }

        if (!response.ok) throw new Error(data.error || 'Request failed.');
        message.textContent = data.message;
        if (endpoint.endsWith('/register')) {
            window.location.href = '/login';
        } else {
            window.location.href = getReturnPath();
        }
    } catch (error) {
        message.textContent = error.message;
        submitButton.disabled = false;
    }
}

showLoginContext();
initializeFirebaseAuth();
