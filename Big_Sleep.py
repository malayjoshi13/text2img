def BigSleep(input_text_prompt):

    from tqdm.notebook import trange
    from IPython.display import Image, display
    from big_sleep import Imagine

    TEXT = input_text_prompt
    SAVE_EVERY = 100 
    SAVE_PROGRESS = True 
    LEARNING_RATE = 5e-2 
    ITERATIONS = 3000 
    SEED = 0 

    model = Imagine(
        text = TEXT,
        save_every = SAVE_EVERY,
        lr = LEARNING_RATE,
        iterations = ITERATIONS,
        save_progress = SAVE_PROGRESS,
        seed = SEED
    )

    for epoch in trange(20, desc = 'epochs'):
        for i in trange(1000, desc = 'iteration'):
            model.train_step(epoch, i)

            if i == 0 or i % model.save_every != 0:
                continue

            filename = TEXT.replace(' ', '_')
            image = Image(f'./{filename}.png')
            display(image)