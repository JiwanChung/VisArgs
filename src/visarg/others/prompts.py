def identification_of_premises_prompt(vp_options, description):
   prompt = f"""
Image:
<image>

Instruction: When given an image, a conclusion, and several visual cue options, you need to identify the visual cue that best relates to the conclusion. To do this effectively, carefully analyze how each visual cue connects to the key elements of the conclusion. Select the visual cue that most directly supports or illustrates the conclusion, ensuring that it enhances the overall understanding and clarity of the message. Answer A), B), or C) with no additional explanation.

Conclusion: {description}
Options:
{vp_options}
ANSWER:"""
   return prompt