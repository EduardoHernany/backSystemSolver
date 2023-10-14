from django.shortcuts import render

from rest_framework import generics
from rest_framework.response import Response
from .models import Matrix
from .serializers import MatrixSerializer
import numpy as np

from sympy import init_printing
init_printing(use_latex='png', scale=1.05, order='grlex',
              forecolor='Black', backcolor='White', fontsize=10)

from sympy import diff, Symbol

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
            f = f'A({j + 1},{i + 1})/pivor = {M[j, i]}/{pivot} = {factor}'
            M[j, :] -= factor * M[i, :]
           
            step = {
                'pivot': pivot,
                'factor': f,
                'step_number': len(matrices),
                'operation': f"L{j + 1} = L{j + 1} - L{i + 1} * {factor:.2f}",
            }
            steps.append(step)

    matrices.append(M.copy())

    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - np.dot(M[i, i + 1:n], x[i + 1:])) / M[i, i]

    return x, M, steps, matrices

    

#-----------------------------------Jordan-----------------------------
def elimination_jordan(M):
    n = M.shape[0]
    x = np.zeros(n)

    matrices = []
    steps = []

    # Passo 1: Aplicar a eliminação de Gauss
    for i in range(n):
        pivot = M[i, i]
        if pivot == 0:
            return None  # Verificar divisão por zero

        for j in range(i + 1, n):
            lj = M[j, :].copy()  
            matrices.append(M.copy())
            factor = M[j, i] / pivot
            f = f'A({j + 1},{i + 1})/pivor = {M[j, i]}/{pivot} = {factor}'
            M[j, :] -= factor * M[i, :]

            step = {
                'pivot': pivot,
                'factor': f,
                'step_number': len(matrices),
                'operation': f"L{j + 1} = L{j + 1} - L{i + 1} * {factor:.2f}",
            }
            steps.append(step)

    matrices.append(M.copy())

    # Passo 2: Aplicar a eliminação de Jordan
    for i in range(n):
        pivot = M[i, i]
        for j in range(i - 1, -1, -1):
            factor = M[j, i] / pivot
            M[j, :] -= factor * M[i, :]
            matrices.append(M.copy())
            step = {
                'factor': factor,
                'step_number': len(matrices),
                'pivot': (pivot),
                'operation': f"L{j + 1} = L{j + 1} - L{i + 1} * {factor:.2f}",
            }
            steps.append(step)

    # Passo 3: Normalizar as linhas
    for i in range(n):
        matrices.append(M.copy())
        step = {
            
            'factor':  M[i, i],
            'step_number': len(matrices),
            'operation': f"L{i + 1} = L{i + 1} / {M[i, i]:.2f}",
        }
        M[i, :] /= M[i, i]
        steps.append(step)

    return M, steps, matrices



def jordan_matrix(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            matrix = data.get('data')

            if matrix:
                matrix = np.array(matrix)

                b = matrix[:, -1]
                
                A = matrix[:, :-1]

                A = A.astype('float64')
                b = b.astype('float64')
                M= matrix.astype('float64')
                M, steps, matrices = elimination_jordan(M)

                
                
                augmented_matrix = np.round(M, 2)

                matrices_as_list_of_lists = [matrix.tolist() for matrix in matrices]

                # Retorne a matriz como JSON
                return JsonResponse({ 'matrix': augmented_matrix.tolist(), 'steps': steps, 'matrices': matrices_as_list_of_lists})
            else:
                return JsonResponse({'error': 'Dados de matriz ausentes ou em formato inválido.'}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Falha ao analisar os dados JSON.'}, status=400)
    else:
        return JsonResponse({'error': 'Método não permitido'}, status=405)
    




#---------------------------------LU-----------------------------------------------------------------------------------------------------------------------------------
def lu_factorization(A):
    n = A.shape[0]
    L = np.identity(n)  
    U = A.copy() 
    matricesL = []
    matricesU = []
    steps = []

    matricesU.append(U.copy())
    matricesL.append(L.copy())

    for i in range(n):
        # Verificar se o pivô é zero
        if U[i, i] == 0:
            return None  

        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor                    # Atualizamos a matriz L com o fator
            U[j, i:] -= factor * U[i, i:]       # Atualizamos a matriz U
            matricesU.append(U.copy())
            matricesL.append(L.copy())
            step = {
                'factor': factor,
                'step_number': len(steps) + 1,
                'operation': f"L{j + 1} = L{j + 1} + L{i + 1} * {factor:.2f} ",
            }
            steps.append(step)

    return matricesL, matricesU, L, U, steps



def lu_matrix(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            matrix = data.get('data')

            if matrix:
                matrix = np.array(matrix)

                A = matrix[:, :-1]
                A = A.astype('float64')

                b = matrix[:, -1]
                b = b.astype('float64')
                
                
                matricesL, matricesU, L, U, steps  = lu_factorization(A)

                matrices_as_list_of_listsL = [matrix.tolist() for matrix in matricesL]
                matrices_as_list_of_listsU = [matrix.tolist() for matrix in matricesU]


                

                LU_verification = np.dot(matricesL[-1], matricesU[-1])

                x, y, stepsx, stepsy = solve_lu(L, U, b)

                return JsonResponse({'x': x.tolist(), 'y': y.tolist(), 'stepX': stepsx , 'stepY': stepsy , 'matricesL': matrices_as_list_of_listsL, 'matricesU': matrices_as_list_of_listsU, 'L': L.tolist(), 'U': U.tolist(), 'steps': steps, 'LU_verification' : LU_verification.tolist()})
            else:
                return JsonResponse({'error': 'Dados de matriz ausentes ou em formato inválido.'}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Falha ao analisar os dados JSON.'}, status=400)
    else:
        return JsonResponse({'error': 'Método não permitido'}, status=405)
    


def solve_lu(L, U, b):
    n = L.shape[0]
    y = np.zeros(n)
    x = np.zeros(n)
    stepsx = []
    stepsy = []

    # Substituição progressiva (Ly = b)
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i, j] * y[j]
        y[i] /= L[i, i]

        step = {
            'step_number': len(stepsy) + 1,
            'operation': f"y[{i + 1}] = {b[i]:.2f}"
        }
        if i > 0:
            for j in range(i):
                step['operation'] += f" - ({L[i, j]:.2f}) * y[{j + 1}]"
            step['operation'] += f" / {L[i, i]:.2f}"
        stepsy.append(step)

    # Substituição regressiva (Ux = y)
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]

        step = {
            'step_number': len(stepsx) + 1,
            'operation': f"x[{i + 1}] = y[{i + 1}]"
        }
        if i < n - 1:
            for j in range(i + 1, n):
                step['operation'] += f" - {U[i, j]:.2f} * x[{j + 1}]"
            step['operation'] += f" / {U[i, i]:.2f}"
        stepsx.append(step)

    return x, y, stepsx, stepsy 


#---------------------------------Jacobi-----------------------------------------------------------------------------------------------------------------------------------




def jacobi_matrix(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            matrix = data.get('data')
            num_iterations = data.get('numIterations', 50)  
            tolerance = data.get('tolerance', 0.0001) 

            tolerance = float(tolerance)
            

            if matrix:
                matrix = np.array(matrix)

                A = matrix[:, :-1]
                A = A.astype('float64')

                b = matrix[:, -1]
                b = b.astype('float64')

                
                result = gauss_seidel(A, b, max_iter=num_iterations, tol= tolerance)

                if result['success']:
                     return JsonResponse({'message': 'Convergência bem-sucedida', 'solution': result['solution'], 'steps': result['steps']})
                else:
                    return JsonResponse({'error': result['message']}, status=200)

            else:
                return JsonResponse({'error': 'Dados de matriz ausentes ou em formato inválido.'}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Falha ao analisar os dados JSON.'}, status=400)
    else:
        return JsonResponse({'error': 'Método não permitido'}, status=405)



def gauss_seidel(A, b, tol=0.0001, max_iter=50):
    n = len(A)
    x = np.zeros(n)  
    steps = []  

    for iter in range(max_iter):
        xx= np.round(x, 3)
        step = {
            'iteration': iter,
            'x': xx.tolist(),
            'equations': [],
            'residual': 0.0
        }
        steps.append(step)

        for i in range(n):
            sigma = 0
            equation = f'x{i+1} = ({b[i]}'
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x[j]
                    equation += f' - {A[i, j]} * x{j}'
            x[i] = (b[i] - sigma) / A[i, i]
            equation += f') / {A[i, i]}'
            step['equations'].append(equation)

        residual = np.linalg.norm(np.dot(A, x) - b)
        step['residual'] = np.round(residual, 10)

        if residual < tol:
            step = {
                'iteration': iter + 1,
                'x': x.tolist(),
                'equations': [],
                'residual': np.round(residual, 10)
            }
            steps.append(step)
            return {'success': True, 'solution': x.tolist(), 'steps': steps}

    return {'success': False, 'message': 'O método de Gauss-Seidel não convergiu após o número máximo de iterações.', 'steps': steps}


#---------------------------------Newton-----------------------------------------------------------------------------------------------------------------------------------



def newton_matrix(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            equationsJ = data.get('equationsJ')
            equationsF = data.get('equationsF')
            
            if equationsF and equationsJ:
              
                equations = equationsF.split(',')
                
                def F(x):
                    x, y = x  # Desempacote as variáveis x e y
                    f1 = eval(equations[0].replace("^", "**"), {"x": x, "y": y})
                    f2 = eval(equations[1].replace("^", "**"), {"x": x, "y": y})
                    return [f1, f2]
                
                equations = equationsJ.split(',')
                
                def J(x):
                    x, y = x  # Desempacote as variáveis x e y
                    f1 = eval(equations[0].replace("^", "**"), {"x": x, "y": y})
                    f2 = eval(equations[1].replace("^", "**"), {"x": x, "y": y})
                    f3 = eval(equations[2].replace("^", "**"), {"x": x, "y": y})
                    f4 = eval(equations[3].replace("^", "**"), {"x": x, "y": y})

                    

                    return [[f1,f2][f3,f4]]
                

                # Valor inicial x0
                x0 = [1.0, 1.0]
                F(x0)

                
                
                return JsonResponse({'message': 'Solução encontrada'})
                
            else:
                return JsonResponse({'error': 'Dados de matriz ou equações ausentes ou em formato inválido.'}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Falha ao analisar os dados JSON.'}, status=400)
    else:
        return JsonResponse({'error': 'Método não permitido'}, status=405)
    



def newton_system(F, x0, matrixJ, tol=1e-6, max_iter=100):
    x = x0
    n = len(x0)
    
    for iter in range(max_iter):
        Fx = F(x)
        Jx = matrixJ
        
        # Solve Jx * delta_x = -Fx for delta_x
        delta_x = np.linalg.solve(Jx, -Fx)

        x = x + delta_x

        if np.linalg.norm(delta_x) < tol:
            return x, iter

    return x, max_iter






