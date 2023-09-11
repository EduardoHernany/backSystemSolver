from django.db import models

# Create your models here.

class Matrix(models.Model):
    rows = models.IntegerField()
    cols = models.IntegerField()
    data = models.JSONField()

    

    class Meta:
        db_table = 'Matrix'  # Especifique o nome da tabela

