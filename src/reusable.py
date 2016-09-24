class Regenerator:

  """
  Regenerator allows generator objects to be reusable.
  Refer to http://stackoverflow.com/questions/1376438/how-to-make-a-repeating-generator-in-python
  """

  def __init__(self, func):
    self.args = ()
    self.kwargs = {}
    self.func = func

  def __call__(self, *args, **kwargs):
    self.args = args
    self.kwargs = kwargs
    return self

  def __iter__(self):
    return self.func(*self.args, **self.kwargs)
