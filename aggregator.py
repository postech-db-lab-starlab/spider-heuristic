class CorrectRatio:
    def __init__(self):
        self.count = 0
        self.correct_count = 0

    def update(self, correct):
        self.count += 1
        if correct:
            self.correct_count += 1

    def get_ratio(self):
        return self.correct_count / self.count

