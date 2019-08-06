import Augmentor

p = Augmentor.Pipeline('D:\\Analysis\\Project\\Image\\Classification\\Flower\\Data\\daisy')

p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
p.sample(2000)
# p.sample(5000)
p.process()

p = Augmentor.Pipeline('D:\\Analysis\\Project\\Image\\Classification\\Flower\\Data\\dandelion')

p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
p.sample(2000)
# p.sample(5000)
p.process()


p = Augmentor.Pipeline('D:\\Analysis\\Project\\Image\\Classification\\Flower\\Data\\rose')

p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
p.sample(5000)
#p.sample(2000)
p.process()


p = Augmentor.Pipeline('D:\\Analysis\\Project\\Image\\Classification\\Flower\\Data\\sunflower')

p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
p.sample(2000)
# p.sample(5000)
p.process()


p = Augmentor.Pipeline('D:\\Analysis\\Project\\Image\\Classification\\Flower\\Data\\tulip')

p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
p.sample(2000)
# p.sample(5000)
p.process()
