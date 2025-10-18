from manim import *
class CNN(Scene):
    def construct(self):
        text = Text("Convolution → ReLU → Pooling → FC Layer")
        self.play(Write(text))
        self.wait(2)
