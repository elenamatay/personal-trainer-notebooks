# Import packages - Note "google-cloud-aiplatform" should be installed and upgraded
import base64
import os
import json
from IPython.display import HTML
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models
from google.cloud import storage


# Available Gemini models
gemini_latest = "gemini-1.5-pro-latest" # Latest version, only for testing or prototyping
gemini_latest_stable = "gemini-1.5-pro" # Latest stable version, ready for prod
gemini_flash_latest_stable = "gemini-1.5-flash"


# Low-bar safety filters
safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}


# Helper functions
def load_video_part(gcs_uri):
    """Loads a video from a URI and creates a corresponding Part object.
    Args:
        uri (str): The GCS URI of the video.
    Returns:
        Part: The loaded video as a Part object.
    """
    file_extension = gcs_uri.split('.')[-1]  # Get the file extension
    mime_type = {
        'mov': "video/quicktime",
        'mp4': "video/mp4"
    }.get(file_extension)

    if mime_type is None:
        raise ValueError("Unsupported video format. Only .mov and .mp4 are currently supported.")

    return Part.from_uri(mime_type=mime_type, uri=gcs_uri)


def exercise_text_info(exercise):
    """Fetches and returns information from the exercises.json file."""
    with open('./exercises.json', 'r') as file:  # Assuming JSON is in the same directory
        data = json.load(file)

    exercise_data = data.get(exercise)

    if exercise_data is None:
        return f"Exercise {exercise} not found in exercises.json"

    text_output = ""

    # Iterate over main keys (e.g., "form", "tempo")
    for main_key, main_value in exercise_data.items():
        text_output += f"\n{main_key}:\n"
        if isinstance(main_value, dict): # CHECK IF IT IS A DICT FIRST
            # Iterate over nested items (e.g., "Steps", "Form" under "form")
            for inner_key, inner_value in main_value.items():
                if isinstance(inner_value, str):
                    formatted_value = inner_value.replace('\\n', '\n')
                    text_output += f"  {inner_key}: {formatted_value}\n"
        else:  # Handle the case where main_value is not a dictionary (e.g., a string)
            text_output += f"  {main_value}\n" # Add string value directly to output

    return text_output


# Call 1: LLM to classify exercise

# Array with all exercises indexes (for classification)
exercises_index_list = """["band-assisted_bench_press", "bar_dip", "bench_press", "bench_press_against_band", "board_press", "cable_chest_press", "close-grip_bench_press", "close-grip_feet-up_bench_press", "decline_bench_press", "decline_push-up", "dumbbell_chest_fly", "dumbbell_chest_press", "dumbbell_decline_chest_press", "dumbbell_floor_press", "dumbbell_pullover", "feet-up_bench_press", "floor_press", "incline_bench_press", "incline_dumbbell_press", "incline_push-up", "kettlebell_floor_press", "kneeling_incline_push-up", "kneeling_push-up", "machine_chest_fly", "machine_chest_press", "pec_deck", "pin_bench_press", "push-up", "push-up_against_wall", "push-ups_with_feet_in_rings", "resistance_band_chest_fly", "smith_machine_bench_press", "smith_machine_incline_bench_press", "standing_cable_chest_fly", "standing_resistance_band_chest_fly", "band_external_shoulder_rotation", "band_internal_shoulder_rotation", "band_pull-apart", "barbell_front_raise", "barbell_rear_delt_row", "barbell_upright_row", "behind_the_neck_press", "cable_lateral_raise", "cable_rear_delt_row", "dumbbell_front_raise", "dumbbell_horizontal_internal_shoulder_rotation", "dumbbell_horizontal_external_shoulder_rotation", "dumbbell_lateral_raise", "dumbbell_rear_delt_row", "dumbbell_shoulder_press", "face_pull", "front_hold", "lying_dumbbell_external_shoulder_rotation", "lying_dumbbell_internal_shoulder_rotation", "machine_lateral_raise", "machine_shoulder_press", "monkey_row", "overhead_press", "plate_front_raise", "power_jerk", "push_press", "reverse_cable_flyes", "reverse_dumbbell_flyes", "reverse_machine_fly", "seated_dumbbell_shoulder_press", "seated_barbell_overhead_press", "seated_smith_machine_shoulder_press", "snatch_grip_behind_the_neck_press", "squat_jerk", "split_jerk", "barbell_curl", "barbell_preacher_curl", "bodyweight_curl", "cable_curl_with_bar", "cable_curl_with_rope", "concentration_curl", "dumbbell_curl", "dumbbell_preacher_curl", "hammer_curl", "incline_dumbbell_curl", "machine_bicep_curl", "spider_curl", "barbell_standing_triceps_extension", "barbell_lying_triceps_extension", "bench_dip", "close-grip_push-up", "dumbbell_lying_triceps_extension", "dumbbell_standing_triceps_extension", "overhead_cable_triceps_extension", "tricep_bodyweight_extension", "tricep_pushdown_with_bar", "tricep_pushdown_with_rope", "air_squat", "barbell_hack_squat", "barbell_lunge", "barbell_walking_lunge", "belt_squat", "body_weight_lunge", "bodyweight_leg_curl", "box_squat", "bulgarian_split_squat", "chair_squat", "dumbbell_lunge", "dumbbell_squat", "front_squat", "goblet_squat", "hack_squat_machine", "half_air_squat", "hip_adduction_machine", "jumping_lunge", "landmine_hack_squat", "landmine_squat", "leg_curl_on_ball", "leg_extension", "leg_press", "lying_leg_curl", "nordic_hamstring_eccentric", "pause_squat", "reverse_barbell_lunge", "romanian_deadlift", "safety_bar_squat", "seated_leg_curl", "shallow_body_weight_lunge", "side_lunges_(bodyweight)", "smith_machine_squat", "squat", "step_up", "zercher_squat", "assisted_chin-up", "assisted_pull-up", "back_extension", "banded_muscle-up", "barbell_row", "barbell_shrug", "block_clean", "block_snatch", "cable_close_grip_seated_row", "cable_wide_grip_seated_row", "chin-up", "clean", "clean_and_jerk", "deadlift", "deficit_deadlift", "dumbbell_deadlift", "dumbbell_row", "dumbbell_shrug", "floor_back_extension", "good_morning", "hang_clean", "hang_power_clean", "hang_power_snatch", "hang_snatch", "inverted_row", "inverted_row_with_underhand_grip", "jefferson_curl", "jumping_muscle-up", "kettlebell_swing", "lat_pulldown_with_pronated_grip", "lat_pulldown_with_supinated_grip", "muscle-up_(bar)", "muscle-up_(rings)", "one-handed_cable_row", "one-handed_lat_pulldown", "pause_deadlift", "pendlay_row", "power_clean", "power_snatch", "pull-up", "pull-up_with_a_neutral_grip", "rack_pull", "ring_pull-up", "ring_row", "seal_row", "seated_machine_row", "snatch", "snatch_grip_deadlift", "stiff-legged_deadlift", "straight_arm_lat_pulldown", "sumo_deadlift", "t-bar_row", "trap_bar_deadlift_with_high_handles", "trap_bar_deadlift_with_low_handles", "banded_side_kicks", "cable_pull_through", "clamshells", "dumbbell_romanian_deadlift", "dumbbell_frog_pumps", "fire_hydrants", "frog_pumps", "glute_bridge", "hip_abduction_against_band", "hip_abduction_machine", "hip_thrust", "hip_thrust_machine", "hip_thrust_with_band_around_knees", "lateral_walk_with_band", "machine_glute_kickbacks", "one-legged_glute_bridge", "one-legged_hip_thrust", "reverse_hyperextension", "romanian_deadlift", "single_leg_romanian_deadlift", "standing_glute_kickback_in_machine", "step_up", "ball_slams", "cable_crunch", "crunch", "dead_bug", "hanging_knee_raise", "hanging_leg_raise", "hanging_sit-up", "high_to_low_wood_chop_with_band", "horizontal_wood_chop_with_band", "kneeling_ab_wheel_roll-out", "kneeling_plank", "kneeling_side_plank", "lying_leg_raise", "lying_windshield_wiper", "lying_windshield_wiper_with_bent_knees", "machine_crunch", "mountain_climbers", "oblique_crunch", "oblique_sit-up", "plank", "plank_with_leg_lifts", "side_plank", "sit-up", "barbell_standing_calf_raise", "eccentric_heel_drop", "heel_raise", "seated_calf_raise", "standing_calf_raise", "barbell_wrist_curl", "barbell_wrist_curl_behind_the_back", "bar_hang", "dumbbell_wrist_curl", "farmers_walk", "fat_bar_deadlift", "gripper", "one-handed_bar_hang", "plate_pinch", "plate_wrist_curl", "towel_pull-up", "barbell_wrist_extension", "dumbbell_wrist_extension", "rowing_machine", "stationary_bike"]"""

# Model configuration
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0.1,
    "top_p": 0.95,
}

# Call definition
def classify_exercise(user_video_uri):
    vertexai.init(project="gymally-test", location="europe-west4")
    model = GenerativeModel(gemini_latest_stable)
    responses = model.generate_content(
        [
            """
            You're an AI Coach. You need to identify the exercise that the user is performing:
            """,
            load_video_part(user_video_uri),
            """
            and classify it as one of the exercise indexes in this list:
            """,
            exercises_index_list,
            """
            Respond only with the exercise index, followed by a line break:
            """
        ],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    classified_exercise = "".join(response.text.strip() for response in responses)
    classified_exercise = classified_exercise.strip('"')

    print(f"Classified exercise: {classified_exercise}\n")
    return classified_exercise


# Call 2: Multimodal model to do the magic and retrieve final response
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0.0,
    "top_p": 0.95,
    "response_mime_type":"application/json",
}

def generate(user_video_uri, classified_exercise):
    vertexai.init(project="gynm-elena", location="europe-west4")
    model = GenerativeModel(gemini_latest_stable)
    responses = model.generate_content(
        [f"""
        You're an AI Coach, an AI-based software that gives corrections or suggestions on how a user performs a strength exercise.
        You need to follow a number of steps in order to give scores and potentially suggest improvements to the user.
        Speak directly to the user in the second person, always in a friendly and motivational tone.
        
        1. The user is performing
        """
        , classified_exercise,
        """
         exercise. This is their video:
        """
        , load_video_part(user_video_uri),
        """
        2. Now, this is all the reference information you have, as an expert coach, on that exercise:
        """
        , exercise_text_info(classified_exercise),
        """
        Note on "tempo": Tempo is a way to describe the speed of each phase of a repetition. It's usually written as a four-digit number (e.g., 2010). Here's what each digit represents:
            - 1st digit: Eccentric phase (lowering the weight)
            - 2nd digit: Pause at the bottom of the movement
            - 3rd digit: Concentric phase (lifting the weight)
            - 4th digit: Pause at the top of the movement
        
        
        3. Compare the user exercise video with the reference content based on the former information. You should do two things with that comparison:
        3.1. Scores. Give a score and suggest improvements to the user as precisely as posibble, taking into account that the reference content includes the right way to perform the exercise.
        Score is a % where 0% means that the user is not following the theory at all, and a 100% means that the user is covering that category perfectly.
        - Give 100% if the user is doing perfect in that category (form, tempo, or range of movement) and you don't find any improvements.
        - Also don't be afraid of giving lower (even really low) scores when needed.
        As part of each score, also suggest all aspects that the user could improve on for each category.
        Finally, give pro tips and and (when applicable) improvements suggestions with all the reference content you have on that exercises.
        
        3.2. Pro tips. Two kinds of "pro tips" you should give to the user in a friendly way:
        
        3.2.1. Detect which reference information can be useful to the user, based on their execution. In an objective while friendly way, let them know things that may be interesting for them.
        These pro tips could be related to potential variations they could do on the form, tempo, range of movement depending on their objectives, or any other aspect that may seem interesting.
        
        3.2.2. (Only if you detect decreases or deviations that the user could having throughout the video, between reps of this same serie - Prioritise this pro tip if so). 
        You could detect changes in the velocity that they are performing (e.g. if they are slowing down each rep), changes in their range of movement, or in their form.
        Objective of this section is trying to spot potential signs of fatigue. While there's research on fatigue's impact on performance, identifying precise visual cues for each individual remains a challenge.
        We do know that fatigue can lead to decreased force production, motor control issues, and an increased risk of injury. Visually, this might manifest as:
        - Tempo: Slowing down of movement, especially in the concentric (lifting) phase.
        - Range of Movement: Decreases in the full range of motion, possibly due to compensatory movements.
        - Form Breakdown: Deviation from proper technique, such as arching the back during squats or leaning forward during rows.
        There's significant individual variation in how fatigue manifests. Factors like training experience, fitness level, and exercise selection play a role. This makes establishing universal visual cues difficult.
        While not foolproof, a potential sign to look out for is with general Observation: Noticeable changes in movement quality, speed, and form compared to earlier repetitions.
        Consequences of Fatigue: Pushing through fatigue can increase the risk of injury, decrease performance, and hinder recovery. It's crucial to recognize the signs and adjust accordingly.
        When to Stop or Adjust: There's no one-size-fits-all answer, but general guidelines include:
        - Form Breakdown: If proper form can't be maintained, stop the set or reduce the weight.
        - Pain: Any sharp or unusual pain warrants immediate cessation of the exercise.
        - Self-Assessment: Listen to your body. If you feel excessively fatigued, stop or modify the workout.
        Alternatives to Continuing:
        - Lower the weight: Use a lighter load that allows for proper form and a full range of motion.
        - Change exercises: Switch to a variation that targets the same muscle groups but places less stress on fatigued areas.
        - Rest: Take a short break to allow for recovery before continuing (e.g. split into more series with less reps).
        If you indentify that a user is having any of the fatigue potential manifestations, ask them to do an exercise of self-Reporting:
        If the user reports feeling fatigued or struggling to maintain proper form throughout th exercise, let them know when to stop or adjust, and alternatives to continuing.
        
        Output should be a valid JSON similar to:
            "Scores": {
                "Form": {
                    "Score": "... (%)",
                    "Improvement Suggestions": "..."
                }
                "Tempo": {
                    "Score": "... (%)",
                    "Improvement Suggestions": "..."
                }
                "Range of Movement": {
                    "Score": "... (%)",
                    "Improvement Suggestions": "..."
                }
            
            "Pro tips": {
                "...:": "...",
                "...": "...",
                "...": "..."
             }
        """
        ],
        
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    for response in responses:
        print(response.text, end="")


# Load videos
videos_folder = "gs://genai-elena/multimodal-video/gym-exercises/exercises-videos"
# GCS URIs
abduction_usr1_20240523_1_uri = f"{videos_folder}/abduction-usr1-20240523-1.mp4"
abduction_usr1_20240527_1_uri = f"{videos_folder}/abduction-usr1-20240527-1.mp4"
abduction_usr1_20240527_2_uri = f"{videos_folder}/abduction-usr1-20240527-2.mp4"
abduction_usr2_20240527_1_uri = f"{videos_folder}/abduction-usr2-20240527-1.mp4"
band_assisted_bench_press_ref_strengthlog_uri = f"{videos_folder}/band-assisted_bench_press_ref_strengthlog.mp4"
bar_dips_ref_strengthlog_uri = f"{videos_folder}/bar_dips_ref_strengthlog.mp4"
bench_press_against_band_ref_strengthlog_uri = f"{videos_folder}/bench_press_against_band_ref_strengthlog.mp4"
bench_press_ref_uri = f"{videos_folder}/bench-press-ref.mov"
bench_press_ref_strengthlog_uri = f"{videos_folder}/strengthlog/bench_press.mp4"
bench_press_usr3_20220501_1_uri = f"{videos_folder}/bench-press-usr3-20220501-ex1.mp4"
bench_press_usr4_20240527_1_uri = f"{videos_folder}/bench-press-usr4-20240527-1.mp4"
bench_press_usr4_20240527_2_uri = f"{videos_folder}/bench-press-usr4-20240527-2.mp4"
board_bench_press_ref_strengthlog_uri = f"{videos_folder}/board_bench_press_ref_strengthlog.mp4"
bulgarian_squad_screen_rec_wrong = f"{videos_folder}/Screen Recording 2024-06-14 at 20.09.41.mov"
cable_chest_press_ref_strengthlog_uri = f"{videos_folder}/cable_chest_press_ref_strengthlog.mp4"
deadlift_usr1_20240626_1_uri = f"{videos_folder}/deadlift_usr1_20240626_1.mp4"
deadlift_usr1_20240627_1_uri = f"{videos_folder}/deadlift_usr1_20240627_1.mp4"
deadlift_usr1_20240627_2_uri = f"{videos_folder}/deadlift_usr1_20240627_2.mp4"
deadlift_usr2_20240627_1_uri = f"{videos_folder}/deadlift_usr2_20240627_1.mp4"
deadlift_usr349283_20240626_uri = f"{videos_folder}/deadlift-usr349283-20240626.mov"
hip_thrust_ref_uri = f"{videos_folder}/hip-thrust-ref.mov"
hip_thrust_usr1_20240501_1_uri = f"{videos_folder}/hip-thrust-usr1-20240501-1.mp4"
hip_thrust_usr1_20240516_1_uri = f"{videos_folder}/hip-thrust-usr1-20240516-1.mp4"
hip_thrust_usr1_20240523_1_uri = f"{videos_folder}/hip-thrust-usr1-20240523-1.mp4"
hip_thrust_usr1_20240523_2_uri = f"{videos_folder}/hip-thrust-usr1-20240523-2.mp4"
hip_thrust_usr2_20240501_1_uri = f"{videos_folder}/hip-thrust-usr2-20240501-1.mp4"
hip_thrust_usr2_20240523_1_uri = f"{videos_folder}/hip-thrust-usr2-20240523-1.mp4"
hip_thrust_usr2_20240523_2_uri = f"{videos_folder}/hip-thrust-usr2-20240523-2.mp4"
leg_press_ref_uri = f"{videos_folder}/leg-press-ref.mp4"
leg_press_usr1_20240501_1_uri = f"{videos_folder}/leg-press-usr1-20240501-1.mp4"
leg_press_usr1_20240523_1_uri = f"{videos_folder}/leg-press-usr1-20240523-1.mp4"
leg_press_usr1_20240523_2_uri = f"{videos_folder}/leg-press-usr1-20240523-2.mp4"
leg_press_usr1_20240530_1_uri = f"{videos_folder}/leg-press-usr1-20240530-1.mp4"
unknown_usr2_20240509_1_uri = f"{videos_folder}/unknown-exercise-usr2-20240509-1.mp4"


# Define user video URI
user_video_uri = bench_press_usr4_20240527_1_uri


# Call models

# Call 1
classified_exercise = classify_exercise(user_video_uri)

# Call 2 
generate(user_video_uri)


# Tokens count and price calculation - Price calculated based on 22Jun: USD 3.50 per 1M input tokens and USD 10.50 per 1M output tokens (for prompts up to 128K tokens). Updated prices in https://ai.google.dev/pricing
model = GenerativeModel(gemini_latest_stable)

# Call 1 tokens - Input
token_count_1_input = model.count_tokens(f"""
        You're an AI Coach. You need to identify the exercise that the user is performing:
        {load_video_part(user_video_uri)}
        and classify it as one of the exercise indexes in this list:
        {exercises_index_list}
        Respond only with the exercise index, followed by a line break:
        """)
print(f"token_count_1_input: {token_count_1_input}")

# Call 1 tokens - Output
token_count_1_output = model.count_tokens('bench_press')
print(f"token_count_1_output: {token_count_1_output}")

# Call 2 tokens - Input
token_count_2_input = model.count_tokens(f"""
        You're an AI Coach, an AI-based software that gives corrections or suggestions on how a user performs a strength exercise.
        You need to follow a number of steps in order to give scores and potentially suggest improvements to the user.
        1. The user is performing {classified_exercise} exercise. This is their video:
        {load_video_part(user_video_uri)}
        2. Now, this is all the reference information you have, as an expert coach, on that exercise:
        {exercise_text_info(classified_exercise)}
        {load_video_part(f"{videos_folder}/strengthlog/{classified_exercise}.mp4")}

        Note on "tempo": Tempo is a way to describe the speed of each phase of a repetition. It's usually written as a four-digit number (e.g., 2010). Here's what each digit represents:
            - 1st digit: Eccentric phase (lowering the weight)
            - 2nd digit: Pause at the bottom of the movement
            - 3rd digit: Concentric phase (lifting the weight)
            - 4th digit: Pause at the top of the movement
        
        
        3. Based on the former information, compare the user exercise video with the reference content. 
        
        In a friendly and motivational way, you should do two things:
        1. Scores. Give a score and suggest improvements to the user as precisely as posibble, taking into account that the reference content includes the right way to perform the exercise.
        Score is a % where 0% means that the user is not following the theory at all, and a 100% means that the user is covering that category perfectly.
        Don't be afraid of giving high-scores: Give 100% if the user is doing perfect in that category (form, tempo, or range of movement).
        Then, give pro tips and and (when applicable) improvements suggestions with all the reference content you have on that exercises.
        
        2. Pro tips. Detect all the reference information you have on that exercise that may be useful to the user based on their execution. 
        In an objective way, let them know all the aspects of the exercise that may be interesting for them. 
        These tips could be related to potential aspects the user could experiment on the exercise form, tempo, range of movement depending on their objectives.
        They could also be related to any other aspect or exercise variation that may seem interesting.
        
        Output should be a valid JSON similar to:
            "Scores":
                "Form":
                    "Score": "... (%)",
                    "Improvement Suggestions": "..."
                "Tempo": 
                    "Score": "... (%)",
                    "Improvement Suggestions": "..."
                "Range of Movement": 
                    "Score": "... (%)",
                    "Improvement Suggestions": "..."
            
            "Pro tips": "..."
        """)
print(f"token_count_2_input: {token_count_2_input}")

# Call 2 tokens - Output
token_count_2_output = model.count_tokens("""
{"Scores": {"Form": {"Score": "90%", "Improvement Suggestions": "The user demonstrates good form overall, but could benefit from ensuring their feet remain flat on the floor throughout the entire lift for optimal stability and power."}, "Tempo": {"Score": "75%", "Improvement Suggestions": "The user could benefit from slowing down the eccentric (lowering) phase of the lift. Aim for a controlled 2-3 second descent to increase muscle activation and improve form."}, "Range of Movement": {"Score": "100%", "Improvement Suggestions": "The user demonstrates a full range of motion, lowering the bar to the chest and pressing it back up to full extension."}}, "Pro tips": "Based on your current form and execution, you might find these tips helpful:\n\n* **Leg Drive:**  You can enhance your stability and potentially lift more weight by incorporating leg drive. Remember to push horizontally with your feet, keeping them flat on the floor, as if trying to slide towards the top of the bench. Avoid pushing your feet downwards.\n* **Grip Width Experimentation:**  Explore different grip widths to target different muscle groups. A wider grip emphasizes the chest and front delts, while a narrower grip recruits more triceps.\n* **Pause Reps:** Consider adding pause reps to your routine. Pausing at the bottom of the movement can help improve technique, strength off the chest, and overall control.\n* **Accessory Exercises:** Incorporate push-ups into your workouts as a valuable accessory exercise. Push-ups engage the same muscle groups as the bench press and can contribute to overall chest and triceps development."}
""")
print(f"token_count_2_output: {token_count_2_output}")

# Price calculation
inputs_price = (token_count_1_input.total_tokens + token_count_2_input.total_tokens) * 3.5 / 1000000
outputs_price = (token_count_1_output.total_tokens + token_count_2_output.total_tokens) * 10.5 / 1000000
total_price = inputs_price + outputs_price

print(f"Price is {inputs_price} due to input tokens + {outputs_price} due to output tokens.")
print(f"Total price: {total_price} USD")