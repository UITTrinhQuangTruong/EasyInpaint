class Clicker:
    def __init__(self, x, y, positive=False, indx=None) -> None:
        self.x = x
        self.y = y

        self.positive = positive 
        self.indx = indx
        
    def __str__(self) -> str:
        return f"x: {self.x}, y: {self.y}, positive: {self.positive}, index: {self.indx}"

    @property
    def is_positive(self):
        return self.positive
    @property
    def value(self):
        return (self.y, self.x, self.indx)