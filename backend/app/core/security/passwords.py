"""
Password hashing and verification using bcrypt.
"""
from passlib.context import CryptContext

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password.
    
    Args:
        plain_password: The plain text password
        hashed_password: The hashed password from database
        
    Returns:
        True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    """
    Hash a plain text password.
    
    Args:
        password: The plain text password
        
    Returns:
        The hashed password
    """
    return pwd_context.hash(password)


def check_password_strength(password: str) -> dict:
    """
    Check password strength and return feedback.
    
    Args:
        password: The password to check
        
    Returns:
        Dictionary with strength score and feedback
    """
    score = 0
    feedback = []
    
    # Length check
    if len(password) >= 8:
        score += 1
    else:
        feedback.append("Password must be at least 8 characters long")
    
    if len(password) >= 12:
        score += 1
    
    # Uppercase check
    if any(c.isupper() for c in password):
        score += 1
    else:
        feedback.append("Add uppercase letters")
    
    # Lowercase check
    if any(c.islower() for c in password):
        score += 1
    else:
        feedback.append("Add lowercase letters")
    
    # Digit check
    if any(c.isdigit() for c in password):
        score += 1
    else:
        feedback.append("Add numbers")
    
    # Special character check
    if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        score += 1
    else:
        feedback.append("Add special characters")
    
    # Determine strength label
    if score >= 6:
        strength = "strong"
    elif score >= 4:
        strength = "medium"
    else:
        strength = "weak"
    
    return {
        "score": score,
        "max_score": 6,
        "strength": strength,
        "feedback": feedback if feedback else ["Password is strong"]
    }
