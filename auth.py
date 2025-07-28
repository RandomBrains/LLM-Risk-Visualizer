"""
Authentication and User Management module for LLM Risk Visualizer
Supports multi-user access, role-based permissions, and session management
"""

import streamlit as st
import hashlib
import sqlite3
import pandas as pd
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import secrets
import re
import os

class UserRole:
    """User role definitions"""
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    
    @classmethod
    def get_all_roles(cls):
        return [cls.ADMIN, cls.ANALYST, cls.VIEWER]
    
    @classmethod
    def get_permissions(cls, role: str) -> Dict[str, bool]:
        """Get permissions for a specific role"""
        permissions = {
            cls.ADMIN: {
                "view_dashboard": True,
                "export_data": True,
                "manage_users": True,
                "configure_apis": True,
                "view_logs": True,
                "modify_settings": True
            },
            cls.ANALYST: {
                "view_dashboard": True,
                "export_data": True,
                "manage_users": False,
                "configure_apis": False,
                "view_logs": True,
                "modify_settings": False
            },
            cls.VIEWER: {
                "view_dashboard": True,
                "export_data": False,
                "manage_users": False,
                "configure_apis": False,
                "view_logs": False,
                "modify_settings": False
            }
        }
        return permissions.get(role, permissions[cls.VIEWER])

class DatabaseManager:
    """Manage user database operations"""
    
    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize user database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'viewer',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                reset_token TEXT,
                reset_token_expires TIMESTAMP
            )
        ''')
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_token TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Activity logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT NOT NULL,
                details TEXT,
                ip_address TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create default admin user if no users exist
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            self._create_default_admin(cursor)
        
        conn.commit()
        conn.close()
    
    def _create_default_admin(self, cursor):
        """Create default admin user"""
        default_password = "admin123"  # Should be changed on first login
        password_hash = self._hash_password(default_password)
        
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, role)
            VALUES (?, ?, ?, ?)
        ''', ("admin", "admin@example.com", password_hash, UserRole.ADMIN))
    
    def _hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', 
                                          password.encode('utf-8'), 
                                          salt.encode('utf-8'), 
                                          100000)
        return salt + password_hash.hex()
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        salt = password_hash[:32]
        stored_hash = password_hash[32:]
        password_hash_check = hashlib.pbkdf2_hmac('sha256',
                                                password.encode('utf-8'),
                                                salt.encode('utf-8'),
                                                100000)
        return stored_hash == password_hash_check.hex()
    
    def create_user(self, username: str, email: str, password: str, role: str = UserRole.VIEWER) -> bool:
        """Create a new user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            password_hash = self._hash_password(password)
            
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, role)
                VALUES (?, ?, ?, ?)
            ''', (username, email, password_hash, role))
            
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False
        except Exception as e:
            print(f"Error creating user: {e}")
            return False
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user and return user info"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, username, email, password_hash, role, is_active
                FROM users WHERE username = ? AND is_active = 1
            ''', (username,))
            
            user = cursor.fetchone()
            
            if user and self._verify_password(password, user[3]):
                # Update last login
                cursor.execute('''
                    UPDATE users SET last_login = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (user[0],))
                conn.commit()
                
                user_info = {
                    'id': user[0],
                    'username': user[1],
                    'email': user[2],
                    'role': user[4],
                    'permissions': UserRole.get_permissions(user[4])
                }
                
                conn.close()
                return user_info
            
            conn.close()
            return None
        except Exception as e:
            print(f"Authentication error: {e}")
            return None
    
    def get_all_users(self) -> List[Dict]:
        """Get all users (admin only)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, username, email, role, created_at, last_login, is_active
                FROM users ORDER BY created_at DESC
            ''')
            
            users = []
            for row in cursor.fetchall():
                users.append({
                    'id': row[0],
                    'username': row[1],
                    'email': row[2],
                    'role': row[3],
                    'created_at': row[4],
                    'last_login': row[5],
                    'is_active': bool(row[6])
                })
            
            conn.close()
            return users
        except Exception as e:
            print(f"Error fetching users: {e}")
            return []
    
    def update_user_role(self, user_id: int, new_role: str) -> bool:
        """Update user role (admin only)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE users SET role = ? WHERE id = ?
            ''', (new_role, user_id))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error updating user role: {e}")
            return False
    
    def deactivate_user(self, user_id: int) -> bool:
        """Deactivate user account"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE users SET is_active = 0 WHERE id = ?
            ''', (user_id,))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error deactivating user: {e}")
            return False
    
    def log_activity(self, user_id: int, action: str, details: str = "", ip_address: str = ""):
        """Log user activity"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO activity_logs (user_id, action, details, ip_address)
                VALUES (?, ?, ?, ?)
            ''', (user_id, action, details, ip_address))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error logging activity: {e}")

class SessionManager:
    """Manage user sessions"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.secret_key = os.environ.get('JWT_SECRET_KEY', secrets.token_hex(32))
    
    def create_session(self, user_info: Dict) -> str:
        """Create a new session token"""
        try:
            # Create JWT token
            payload = {
                'user_id': user_info['id'],
                'username': user_info['username'],
                'role': user_info['role'],
                'exp': datetime.utcnow() + timedelta(hours=24)
            }
            
            token = jwt.encode(payload, self.secret_key, algorithm='HS256')
            
            # Store session in database
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            expires_at = datetime.utcnow() + timedelta(hours=24)
            cursor.execute('''
                INSERT INTO sessions (user_id, session_token, expires_at)
                VALUES (?, ?, ?)
            ''', (user_info['id'], token, expires_at))
            
            conn.commit()
            conn.close()
            
            return token
        except Exception as e:
            print(f"Error creating session: {e}")
            return ""
    
    def validate_session(self, token: str) -> Optional[Dict]:
        """Validate session token"""
        try:
            # Decode JWT token
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # Check if session exists in database
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_id, expires_at, is_active
                FROM sessions 
                WHERE session_token = ? AND is_active = 1
            ''', (token,))
            
            session = cursor.fetchone()
            conn.close()
            
            if session and datetime.fromisoformat(session[1]) > datetime.utcnow():
                return {
                    'user_id': payload['user_id'],
                    'username': payload['username'],
                    'role': payload['role'],
                    'permissions': UserRole.get_permissions(payload['role'])
                }
            
            return None
        except jwt.ExpiredSignatureError:
            return None
        except Exception as e:
            print(f"Session validation error: {e}")
            return None
    
    def invalidate_session(self, token: str):
        """Invalidate a session"""
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE sessions SET is_active = 0
                WHERE session_token = ?
            ''', (token,))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error invalidating session: {e}")

class AuthManager:
    """Main authentication manager"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.session_manager = SessionManager(self.db)
    
    def login(self, username: str, password: str) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """Login user"""
        # Validate input
        if not self._validate_username(username) or not password:
            return False, "Invalid username or password", None
        
        # Authenticate user
        user_info = self.db.authenticate_user(username, password)
        if not user_info:
            return False, "Invalid username or password", None
        
        # Create session
        token = self.session_manager.create_session(user_info)
        if not token:
            return False, "Failed to create session", None
        
        # Log activity
        self.db.log_activity(user_info['id'], "login", f"User {username} logged in")
        
        return True, token, user_info
    
    def logout(self, token: str):
        """Logout user"""
        self.session_manager.invalidate_session(token)
    
    def register(self, username: str, email: str, password: str, role: str = UserRole.VIEWER) -> Tuple[bool, str]:
        """Register new user"""
        # Validate input
        if not self._validate_username(username):
            return False, "Invalid username format"
        
        if not self._validate_email(email):
            return False, "Invalid email format"
        
        if not self._validate_password(password):
            return False, "Password must be at least 8 characters with uppercase, lowercase, and number"
        
        if role not in UserRole.get_all_roles():
            return False, "Invalid role"
        
        # Create user
        if self.db.create_user(username, email, password, role):
            return True, "User created successfully"
        else:
            return False, "Username or email already exists"
    
    def _validate_username(self, username: str) -> bool:
        """Validate username format"""
        if not username or len(username) < 3 or len(username) > 20:
            return False
        return re.match(r'^[a-zA-Z0-9_]+$', username) is not None
    
    def _validate_email(self, email: str) -> bool:
        """Validate email format"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(email_pattern, email) is not None
    
    def _validate_password(self, password: str) -> bool:
        """Validate password strength"""
        if len(password) < 8:
            return False
        
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        
        return has_upper and has_lower and has_digit

# Streamlit integration functions
def init_auth_state():
    """Initialize authentication state in Streamlit session"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.token = None

def login_form():
    """Display login form"""
    st.title("🔐 Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            auth_manager = AuthManager()
            success, message, user_info = auth_manager.login(username, password)
            
            if success:
                st.session_state.authenticated = True
                st.session_state.user = user_info
                st.session_state.token = message
                st.success("Login successful!")
                st.rerun()
            else:
                st.error(message)

def logout():
    """Logout current user"""
    if st.session_state.get('token'):
        auth_manager = AuthManager()
        auth_manager.logout(st.session_state.token)
    
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.token = None
    st.rerun()

def require_permission(permission: str):
    """Decorator to require specific permission"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not st.session_state.get('authenticated'):
                st.error("Please login to access this feature")
                return
            
            user = st.session_state.get('user', {})
            permissions = user.get('permissions', {})
            
            if not permissions.get(permission, False):
                st.error("You don't have permission to access this feature")
                return
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def get_current_user() -> Optional[Dict]:
    """Get current authenticated user"""
    return st.session_state.get('user')

def is_authenticated() -> bool:
    """Check if user is authenticated"""
    return st.session_state.get('authenticated', False)

def has_permission(permission: str) -> bool:
    """Check if current user has specific permission"""
    if not is_authenticated():
        return False
    
    user = st.session_state.get('user', {})
    permissions = user.get('permissions', {})
    return permissions.get(permission, False)