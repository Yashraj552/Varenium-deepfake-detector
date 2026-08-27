import os
import unittest

from backend.main import app, db, User


class SecuritySecurityTests(unittest.TestCase):
    def setUp(self):
        app.config.update(
            TESTING=True,
            SQLALCHEMY_DATABASE_URI='sqlite:///:memory:',
            SECRET_KEY='test-secret-key',
            SESSION_COOKIE_SECURE=False,
            SESSION_COOKIE_HTTPONLY=True,
            SESSION_COOKIE_SAMESITE='Lax',
            WTF_CSRF_ENABLED=False,
        )
        with app.app_context():
            db.drop_all()
            db.create_all()

    def test_login_rate_limit_blocks_repeated_failures(self):
        with app.test_client() as client:
            for _ in range(5):
                resp = client.post('/auth/login', json={'email': 'attacker@example.com', 'password': 'wrong-pass'})
                if resp.status_code == 429:
                    break

            resp = client.post('/auth/login', json={'email': 'attacker@example.com', 'password': 'wrong-pass'})
            self.assertEqual(resp.status_code, 429, resp.get_data(as_text=True))


if __name__ == '__main__':
    unittest.main()
