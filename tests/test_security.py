import os
import unittest
import bcrypt

from backend.main import app, db, AuditLog, Investigation, User


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

    def test_successful_login_is_audited(self):
        password_hash = bcrypt.hashpw(b'correct-password', bcrypt.gensalt()).decode('utf-8')
        with app.app_context():
            db.session.add(User(
                name='Test User',
                email='user@example.com',
                password_hash=password_hash,
                role='user',
                is_active=True,
            ))
            db.session.commit()

        with app.test_client() as client:
            resp = client.post('/auth/login', json={
                'email': 'user@example.com',
                'password': 'correct-password',
            })
            self.assertEqual(resp.status_code, 200)

        with app.app_context():
            audit = AuditLog.query.filter_by(action='USER_LOGIN', status='SUCCESS').first()
            self.assertIsNotNone(audit)

    def test_missing_route_returns_not_found(self):
        with app.test_client() as client:
            resp = client.get('/does-not-exist')
            self.assertEqual(resp.status_code, 404)

    def test_upload_access_is_limited_to_owner(self):
        with app.app_context():
            owner = User(name='Owner', email='owner@example.com', password_hash='unused', role='user')
            other = User(name='Other', email='other@example.com', password_hash='unused', role='user')
            db.session.add_all([owner, other])
            db.session.commit()
            db.session.add(Investigation(
                user_id=owner.id,
                original_filename='private.jpg',
                stored_filename='private-file.jpg',
                file_hash='hash',
                image_data=b'private-image',
                analysis_result='AUTHENTIC',
                confidence_score=99.0,
            ))
            db.session.commit()
            other_id = other.id

        with app.test_client() as client:
            with client.session_transaction() as session:
                session['user_id'] = other_id
            resp = client.get('/uploads/private-file.jpg')
            self.assertEqual(resp.status_code, 403)


if __name__ == '__main__':
    unittest.main()
