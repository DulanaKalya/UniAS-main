from manim import *

class UniASArchitecture(Scene):
    def construct(self):
        # Input image
        input_img = Rectangle(color=BLUE, height=2, width=2).shift(LEFT*4)
        input_text = Text("Input Image").scale(0.5).next_to(input_img, DOWN)
        self.play(Create(input_img), Write(input_text))

        # Backbone block
        backbone = Rectangle(color=GREEN, height=2, width=3).next_to(input_img, RIGHT, buff=1)
        backbone_text = Text("EfficientNet Backbone").scale(0.4).next_to(backbone, DOWN)
        self.play(Create(backbone), Write(backbone_text))
        self.play(input_img.animate.shift(RIGHT*1.5), run_time=1)

        # Multi-scale arrows
        for i in range(4):
            arrow = Arrow(backbone.get_right(), RIGHT*3 + UP*(i-1.5))
            self.play(Create(arrow), run_time=0.5)

        # Transformer block
        transformer = Rectangle(color=YELLOW, height=3, width=3).shift(RIGHT*4)
        transformer_text = Text("Transformer / MaskFormer").scale(0.4).next_to(transformer, DOWN)
        self.play(Create(transformer), Write(transformer_text))
