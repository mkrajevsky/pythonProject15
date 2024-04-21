from openai import OpenAI


client = OpenAI(
  organization='org-vgMQtXEn1gQlKz2UIfqcxhex',
  project='proj_ftpHF4vxIslTZ1WVlbHpZX4I',
  api_key='sk-1ZQ6J9Q6Q6J9')


response = client.create_prompt("Once upon a time")