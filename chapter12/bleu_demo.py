import evaluate

predictions = [
    "It is a guide to action which ensures that the military always obeys the commands of the party"
]

references = [
    [
        "It is a guide to action that ensures that the military will forever heed Party commands",
        "It is the guiding principle which guarantees the military forces always being under the command of the Party",
        "It is the practical guide for the army always to heed the directions of the party",
    ]
]

bleu = evaluate.load("bleu")

print("default", bleu.compute(predictions=predictions, references=references))

print("n=1", bleu.compute(predictions=predictions, references=references, max_order=1))

print("n=2", bleu.compute(predictions=predictions, references=references, max_order=2))
