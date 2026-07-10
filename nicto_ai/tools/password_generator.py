"""
NICTO AI - Password Generator Tool
Generate secure passwords with customizable options.
"""

import secrets
import string
import logging
from typing import Dict
from .base import Tool, ToolResult, ToolParameter

logger = logging.getLogger(__name__)


class PasswordGeneratorTool(Tool):
    """
    Secure password generator.

    Features:
    - Customizable length
    - Character type selection
    - Passphrase generation
    - Strength estimation
    - Exclusion patterns
    """

    name = "password_generator"
    description = "Generate secure passwords with customizable length, character types, and strength requirements."
    parameters = [
        ToolParameter(name="length", type="integer", description="Password length (8-128)", required=False, default=16),
        ToolParameter(name="include_uppercase", type="boolean", description="Include uppercase letters", required=False, default=True),
        ToolParameter(name="include_lowercase", type="boolean", description="Include lowercase letters", required=False, default=True),
        ToolParameter(name="include_digits", type="boolean", description="Include digits", required=False, default=True),
        ToolParameter(name="include_symbols", type="boolean", description="Include symbols", required=False, default=True),
        ToolParameter(name="count", type="integer", description="Number of passwords to generate (1-10)", required=False, default=1),
        ToolParameter(name="exclude_chars", type="string", description="Characters to exclude", required=False),
        ToolParameter(name="passphrase", type="boolean", description="Generate passphrase instead", required=False, default=False),
    ]
    tags = ["password", "security", "generator"]
    timeout_seconds = 5.0

    # Common word list for passphrases
    WORDS = [
        "apple", "bridge", "castle", "dragon", "eagle", "forest", "garden",
        "harbor", "island", "jungle", "knight", "lemon", "mountain", "noble",
        "ocean", "palace", "quartz", "river", "shadow", "temple", "umbrella",
        "village", "window", "yellow", "zenith", "autumn", "breeze", "crystal",
        "dream", "ember", "frost", "glow", "haze", "ivory", "jade",
        "karma", "lunar", "mist", "nova", "orbit", "prism", "quest",
    ]

    def _execute(self, length: int = 16, include_uppercase: bool = True,
                 include_lowercase: bool = True, include_digits: bool = True,
                 include_symbols: bool = True, count: int = 1,
                 exclude_chars: str = None, passphrase: bool = False) -> ToolResult:

        length = max(8, min(128, length))
        count = max(1, min(10, count))

        if passphrase:
            passwords = [self._generate_passphrase() for _ in range(count)]
        else:
            passwords = [
                self._generate_password(length, include_uppercase, include_lowercase,
                                       include_digits, include_symbols, exclude_chars)
                for _ in range(count)
            ]

        results = []
        for pwd in passwords:
            strength = self._estimate_strength(pwd)
            results.append({
                "password": pwd,
                "length": len(pwd),
                "strength": strength,
            })

        return ToolResult(
            success=True,
            output={
                "passwords": results if count > 1 else results[0],
                "count": count,
            },
        )

    def _generate_password(self, length: int, uppercase: bool, lowercase: bool,
                          digits: bool, symbols: bool, exclude: str = None) -> str:
        """Generate a random password"""
        chars = ""
        required = []

        if uppercase:
            chars += string.ascii_uppercase
            required.append(secrets.choice(string.ascii_uppercase))
        if lowercase:
            chars += string.ascii_lowercase
            required.append(secrets.choice(string.ascii_lowercase))
        if digits:
            chars += string.digits
            required.append(secrets.choice(string.digits))
        if symbols:
            symbols_str = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            chars += symbols_str
            required.append(secrets.choice(symbols_str))

        if not chars:
            chars = string.ascii_letters + string.digits

        # Remove excluded characters
        if exclude:
            chars = ''.join(c for c in chars if c not in exclude)

        # Generate password
        password = required.copy()
        for _ in range(length - len(required)):
            password.append(secrets.choice(chars))

        # Shuffle
        password_list = list(password)
        for i in range(len(password_list) - 1, 0, -1):
            j = secrets.randbelow(i + 1)
            password_list[i], password_list[j] = password_list[j], password_list[i]

        return ''.join(password_list)

    def _generate_passphrase(self) -> str:
        """Generate a memorable passphrase"""
        words = [secrets.choice(self.WORDS) for _ in range(4)]
        # Capitalize first letter of each word
        words = [w.capitalize() for w in words]
        # Add a number
        number = secrets.randbelow(100)
        return f"{words[0]}{words[1]}{number}{words[2]}{words[3]}"

    def _estimate_strength(self, password: str) -> Dict:
        """Estimate password strength"""
        score = 0
        feedback = []

        # Length
        if len(password) >= 12:
            score += 2
        elif len(password) >= 8:
            score += 1
        else:
            feedback.append("Use at least 8 characters")

        # Character types
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_symbol = any(not c.isalnum() for c in password)

        types_count = sum([has_upper, has_lower, has_digit, has_symbol])
        score += types_count

        if not has_upper:
            feedback.append("Add uppercase letters")
        if not has_lower:
            feedback.append("Add lowercase letters")
        if not has_digit:
            feedback.append("Add numbers")
        if not has_symbol:
            feedback.append("Add symbols")

        # Strength label
        if score >= 6:
            strength = "Very Strong"
        elif score >= 4:
            strength = "Strong"
        elif score >= 3:
            strength = "Medium"
        else:
            strength = "Weak"

        # Entropy estimate
        charset_size = 0
        if has_upper:
            charset_size += 26
        if has_lower:
            charset_size += 26
        if has_digit:
            charset_size += 10
        if has_symbol:
            charset_size += 32
        if charset_size == 0:
            charset_size = 62

        import math
        entropy = len(password) * math.log2(charset_size)

        return {
            "label": strength,
            "score": score,
            "max_score": 7,
            "entropy_bits": round(entropy, 1),
            "feedback": feedback,
        }
