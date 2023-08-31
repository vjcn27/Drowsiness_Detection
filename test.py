class Parent:
    def __init__(self):
        self.num=56
    def show(self):
        print("parent ",self.num)
class Child(Parent):
    def __init__(self):
        super().__init__()
        self.var=86
    def show(self):
        print('Child ',slef.var)
parent=Parent()
parent.show()
child=Child()
child.show()        
