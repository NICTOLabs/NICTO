"""
Emotional Processing Module
Multi-modal emotion detection, empathetic response generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class EmotionalStateModel(nn.Module):
    """
    Emotional State Model (ESM)
    
    Tracks and predicts emotional states using:
    - Valence (positive-negative)
    - Arousal (calm-excited)
    - Dominance (submissive-dominant)
    
    Based on the VAD model of emotions
    """
    
    def __init__(self, dim: int = 4096, emotion_dims: int = 3):
        super().__init__()
        self.dim = dim
        self.emotion_dims = emotion_dims
        
        # Emotion detection from text
        self.emotion_detector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, emotion_dims),
            nn.Tanh(),  # Output in [-1, 1]
        )
        
        # Emotional trajectory predictor
        self.trajectory_predictor = nn.Sequential(
            nn.Linear(dim + emotion_dims, dim),
            nn.SiLU(),
            nn.Linear(dim, emotion_dims),
        )
        
        # Emotional memory
        self.emotion_memory = nn.Parameter(torch.zeros(1, emotion_dims))
        
        # Update gate
        self.update_gate = nn.Linear(dim + emotion_dims, emotion_dims)
    
    def forward(
        self,
        x: torch.Tensor,
        previous_emotion: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Detect and track emotional state
        
        Args:
            x: Input representation [batch_size, seq_len, dim]
            previous_emotion: Previous emotional state [batch_size, emotion_dims]
            
        Returns:
            Dictionary with emotional states
        """
        batch_size = x.shape[0]
        
        # Global representation
        x_global = x.mean(dim=1)  # [B, dim]
        
        # Detect current emotion
        current_emotion = self.emotion_detector(x_global)  # [B, 3]
        
        # Use memory if no previous emotion
        if previous_emotion is None:
            previous_emotion = self.emotion_memory.expand(batch_size, -1)
        
        # Update emotional state with memory
        update_input = torch.cat([x_global, previous_emotion], dim=-1)
        update_gate = torch.sigmoid(self.update_gate(update_input))
        updated_emotion = update_gate * current_emotion + (1 - update_gate) * previous_emotion
        
        # Predict trajectory
        trajectory_input = torch.cat([x_global, updated_emotion], dim=-1)
        trajectory = self.trajectory_predictor(trajectory_input)
        
        return {
            "current_emotion": current_emotion,
            "updated_emotion": updated_emotion,
            "trajectory": trajectory,
            "emotion_memory": self.emotion_memory,
        }


class EmpathyGenerator(nn.Module):
    """
    Empathy Generation System
    
    Generates empathetic responses using three types:
    - Cognitive empathy (understanding)
    - Affective empathy (feeling)
    - Compassionate empathy (helping)
    """
    
    def __init__(self, dim: int = 4096):
        super().__init__()
        self.dim = dim
        
        # Cognitive empathy (understanding perspective)
        self.cognitive = nn.Sequential(
            nn.Linear(dim + 3, dim),  # +3 for emotion
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        
        # Affective empathy (emotional resonance)
        self.affective = nn.Sequential(
            nn.Linear(dim + 3, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        
        # Compassionate empathy (desire to help)
        self.compassionate = nn.Sequential(
            nn.Linear(dim + 3, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        
        # Empathy fusion
        self.fusion = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        
        # Empathy calibration
        self.calibration = nn.Sequential(
            nn.Linear(dim + 3, dim),
            nn.Sigmoid(),
        )
    
    def forward(
        self, x: torch.Tensor, emotion: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Generate empathetic response components
        
        Args:
            x: Input representation [batch_size, seq_len, dim]
            emotion: Emotional state [batch_size, 3]
            
        Returns:
            Dictionary with empathy components
        """
        x_global = x.mean(dim=1)
        
        # Combine input and emotion
        combined = torch.cat([x_global, emotion], dim=-1)
        
        # Generate empathy components
        cognitive = self.cognitive(combined)
        affective = self.affective(combined)
        compassionate = self.compassionate(combined)
        
        # Fuse all empathy types
        empathy_combined = torch.cat([cognitive, affective, compassionate], dim=-1)
        empathy_output = self.fusion(empathy_combined)
        
        # Calibrate empathy
        calibration = self.calibration(combined)
        calibrated_empathy = empathy_output * calibration
        
        return {
            "cognitive_empathy": cognitive,
            "affective_empathy": affective,
            "compassionate_empathy": compassionate,
            "empathy_output": calibrated_empathy,
            "calibration": calibration,
        }


class EmotionalResponseGenerator(nn.Module):
    """
    Complete Emotional Response Generator
    
    Combines:
    - Emotion detection
    - Empathy generation
    - Response calibration
    """
    
    def __init__(self, dim: int = 4096):
        super().__init__()
        self.dim = dim
        
        # Emotional state model
        self.emotion_model = EmotionalStateModel(dim)
        
        # Empathy generator
        self.empathy_generator = EmpathyGenerator(dim)
        
        # Response calibration
        self.response_calibration = nn.Sequential(
            nn.Linear(dim + 3, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        previous_emotion: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate emotionally-aware response
        
        Args:
            x: Input representation
            previous_emotion: Previous emotional state
            
        Returns:
            Dictionary with emotional outputs
        """
        # Detect emotion
        emotion_outputs = self.emotion_model(x, previous_emotion)
        emotion = emotion_outputs["updated_emotion"]
        
        # Generate empathy
        empathy_outputs = self.empathy_generator(x, emotion)
        
        # Calibrate response
        calibration_input = torch.cat([x.mean(dim=1), emotion], dim=-1)
        calibration = self.response_calibration(calibration_input)
        
        # Generate final output
        emotional_output = empathy_outputs["empathy_output"] * calibration
        output = self.output_proj(emotional_output)
        
        return {
            "output": output,
            "emotion": emotion,
            "empathy": empathy_outputs,
            "calibration": calibration,
        }
