import base64

def audio_to_base64(file_path):
    try:
        with open(file_path, "rb") as audio_file:
            # Read binary file and encode to base64 bytes
            encoded_bytes = base64.b64encode(audio_file.read())
            # Decode bytes to string
            return encoded_bytes.decode('utf-8')
    except FileNotFoundError:
        return "File not found."
 
# Usage
file_name = "human_494.mp3"  # Replace with your audio file path
base64_string = audio_to_base64(file_name)

# Print full base64 string
print(base64_string)