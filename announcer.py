import time

class ThreatAnnouncer:
    def __init__(self, frame_width):
        self.frame_width = frame_width
        
        # --- NLP Templates ---
        self.class_templates = {
            'person': 'Person', 'car': 'Car', 'truck': 'Truck',
            'motorcycle': 'Motorcycle', 'bus': 'Bus'
        }
        self.reason_templates = {
            'looming': 'approaching fast',
            'proximity': 'in your path',
            'anomaly': 'moving erratically',
            'contextual': 'detected'
        }
        self.direction_templates = {
            'front': 'in front',
            'front-left': 'to your front-left',
            'front-right': 'to your front-right',
            'left': 'on your left',
            'right': 'on your right'
        }
        
        # --- State for Debouncing ---
        self.last_message = ""
        self.last_message_time = 0
        self.debounce_interval = 3.0  # seconds

    def _get_direction(self, x_center):
        """Calculates the direction of an object based on its x-coordinate."""
        third_width = self.frame_width / 3
        if x_center < third_width:
            return 'left'
        elif x_center > 2 * third_width:
            return 'right'
        else:
            return 'front'
            
    def _prioritize_threats(self, threats):
        """Finds the single most important threat from a list."""
        if not threats:
            return None
        # Simple prioritization: the one with the highest score wins.
        # A more advanced version could have tie-breakers (e.g., looming > proximity).
        return max(threats, key=lambda t: t['threat_score'])

    def generate_message(self, threats):
        """The main function. Takes a list of threat dicts and returns a message string."""
        
        most_urgent_threat = self._prioritize_threats(threats)
        
        if not most_urgent_threat:
            self.last_message = "" # Clear last message if no threats
            return None
            
        # Translate components into words
        obj_name = self.class_templates.get(most_urgent_threat['class_name'], 'Object')
        reason = self.reason_templates.get(most_urgent_threat['primary_reason'], 'detected')
        direction = self._get_direction(most_urgent_threat['x_center'])
        direction_phrase = self.direction_templates.get(direction, 'nearby')

        # Assemble the message
        message = f"Warning: {obj_name} {reason} {direction_phrase}."
        
        # Debouncing logic: Only announce if the message is new or it's been a while
        current_time = time.time()
        if message != self.last_message or (current_time - self.last_message_time > self.debounce_interval):
            self.last_message = message
            self.last_message_time = current_time
            return message
            
        return None # Suppress the message if it's a repeated announcement