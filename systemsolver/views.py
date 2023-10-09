from django.shortcuts import render

from rest_framework import generics
from rest_framework.response import Response
from .models import Matrix
from .serializers import MatrixSerializer
import numpy as np



from django.http import JsonResponse
import json

def jordan_matrix_matrix(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            matrix = data.get('data')

            if matrix:
                matrix = np.array(matrix)

                b = matrix[:, -1]
                # Separe 'A' do restante da matriz
                A = matrix[:, :-1]

                M= matrix.astype('float64')

                x, M, steps, matrices = elimination_gaussianaa(M)

                
                matrices_as_list_of_lists = [matrix.tolist() for matrix in matrices]

                # Retorne a matriz como JSON
                return JsonResponse({'result': x.tolist(), 'matrix': M.tolist(), 'steps': steps, 'matrices': matrices_as_list_of_lists})
            else:
                return JsonResponse({'error': 'Dados de matriz ausentes ou em formato inválido.'}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Falha ao analisar os dados JSON.'}, status=400)
    else:
        return JsonResponse({'error': 'Método não permitido'}, status=405)



def elimination_gaussianaa(M):
    n = M.shape[0]
    x = np.zeros(n)

    matrices = []
    steps = []

    for i in range(n):
        pivot = M[i, i]
        if pivot == 0:
            return None  # Verificar divisão por zero
        
        for j in range(i + 1, n):
            lj = M[j, :].copy()  # Copiar a linha inteira antes da operação
            matrices.append(M.copy())
            factor = M[j, i] / pivot
            f = f'A({j + 1},{i + 1})/pivot = {M[j, i]}/{pivot} = {factor}'
            M[j, :] -= factor * M[i, :]
           
            step = {
                'pivot': pivot,
                'factor': f,
                'step_number': len(matrices),
                'operation': f"R{j + 1} = R{j + 1} - {factor:.2f} * R{i + 1}",
            }
            steps.append(step)

    matrices.append(M.copy())

    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - np.dot(M[i, i + 1:n], x[i + 1:])) / M[i, i]

    return x, M, steps, matrices

    

#-----------------------------------Jordan-----------------------------

def jordan_elimination(A, b):
    n = len(b)
    x = np.zeros(n)

    matrices = []
    steps = []

    # Fase de eliminação
    for i in range(n):
        pivot = A[i, i]
        if pivot == 0:
            return None  # Verificar divisão por zero
        
        for j in range(n):
            if i != j:
                factor = A[j, i] / pivot
                f = f'A({j+1},{i+1})/pivô  = {A[j,i]}/{pivot} = {factor}'
                b[j] -= factor * b[i]
                A[j, i:] -= factor * A[i, i:]
                print( A[j, i:])
                
                step = {
                    'pivo': pivot,
                    'factor': f,
                    'step_number': len(matrices),
                    'operation': f"L{j + 1} =  {factor:.2f} * L{j + 1} - L{i + 1}",
                }
                steps.append(step)

        matrices.append(np.column_stack((A.copy(), b.copy())))

    # Fase de normalização
    for i in range(n):
        factor = A[i, i]
        b[i] /= factor
        A[i, :] /= factor

    matrices.append(np.column_stack((A.copy(), b.copy())))

    return b, A, steps, matrices




def jordan_matrix(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            matrix = data.get('data')

            if matrix:
                matrix = np.array(matrix)

                b = matrix[:, -1]
                # Separe 'A' do restante da matriz
                A = matrix[:, :-1]

                A = A.astype('float64')
                b = b.astype('float64')
                x, A, steps, matrices = jordan_elimination(A, b)

                # Arredonde os elementos da matriz aumentada com duas casas decimais
                augmented_matrix = np.column_stack((A, b))
                augmented_matrix = np.round(augmented_matrix, 2)

                matrices_as_list_of_lists = [matrix.tolist() for matrix in matrices]

                # Retorne a matriz como JSON
                return JsonResponse({'result': x.tolist(), 'matrix': augmented_matrix.tolist(), 'steps': steps, 'matrices': matrices_as_list_of_lists})
            else:
                return JsonResponse({'error': 'Dados de matriz ausentes ou em formato inválido.'}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Falha ao analisar os dados JSON.'}, status=400)
    else:
        return JsonResponse({'error': 'Método não permitido'}, status=405)