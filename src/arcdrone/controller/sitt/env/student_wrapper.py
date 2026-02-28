"""
Purpose:

when call

env = StudentWrapper(teacher_env, student_obs_fn)

It generate a env with modified step and reset function!
I will bring student_obs_fn an script that its use in a real RL env. Because I will train the teacher with RL. 
Then teach the student with IL or distillation and finally do finetuning of the student RL env. So there exist a student RL env!

The goal is to get the student observations. This could be rendered images, so with get_student_obs flag we can decide if we want to compute the student obs or not.
But observations allways have the same shape nonetheless. Therefore we need a boolean valid_student_obs to know if the student obs are valid or not


Pseudo code:

dataclass:
StateforStudent(state):
    obs_student:
    info_student: (like obs_history or like get_student_obs flag. Also valid_student_obs)

StudentWrapper(inherit from teacher_env):


    def __init__(...):
        # initialize the parent
        # inite some constants needed for this script logic

    def step(...)-> StateforStudent:
        state = super().step(...)
        if get_student_obs:
            state_broken = student_obs_fn(state) # This update 
            state = state._replace(obs_student=state_broken.obs_student, info_student=state_broken.info_student)
            info_student = get_info_student(state)
            return state

    def reset(...)-> StateforStudent:
        state = super().reset(...)
        if get_student_obs:
            reset of this variables: obs_student, info_student 
            












"""