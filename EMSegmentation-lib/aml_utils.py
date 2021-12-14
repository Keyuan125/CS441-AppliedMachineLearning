import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import traceback
from pygments import formatters, highlight, lexers
import re
import inspect
import types
import copy

visualize = True
perform_computation = True
test_db_dir = os.path.dirname(os.path.realpath(__file__)) + '/test_db'
NO_DASHES = 55

Fore_BLUE_LIGHT = u'\u001b[38;5;19m'
Fore_RED_LIGHT = u'\u001b[38;5;196m'
Fore_BLUE = u'\u001b[38;5;34m'
Fore_RED = '\x1b[1;31m'
FORE_GREEN_DARK = u'\u001b[38;5;22m'
Fore_DARKRED = u'\u001b[38;5;124m'
Fore_MAGENTA = '\x1b[1m' + u'\u001b[38;5;92m'
Fore_GREEN = u'\u001b[38;5;32m'
Fore_BLACK = '\x1b[0m' + u'\u001b[38;5;0m'


########################################################################################
######################## Utilities for traceback processing ############################
########################################################################################
def keep_tb_rule(tb):
    tb_file_path = tb.tb_frame.f_code.co_filename
    if os.path.realpath(__file__) == os.path.realpath(tb_file_path):
        return False
    else:
        return True

def censor_exc_traceback(exc_traceback):
    original_tb_list = []
    tb_next = exc_traceback
    while tb_next is not None:
        original_tb_list.append(tb_next)
        tb_next = tb_next.tb_next
        
    censored_tb_list = [tb for tb in original_tb_list if keep_tb_rule(tb)]
    
    for i, tb in enumerate(censored_tb_list[:-1]):
        tb.tb_next = censored_tb_list[i+1]
    
    if len(censored_tb_list) > 0:
        return censored_tb_list[0]
    else:
        return exc_traceback

try:
    import IPython
    ultratb = IPython.core.ultratb.VerboseTB(include_vars=False)
    def get_tb_colored_str(exc_type, exc_value, exc_traceback):
        manipulated_exc_traceback = censor_exc_traceback(exc_traceback)
        tb_text = ultratb.text(exc_type, exc_value, manipulated_exc_traceback)
        tb_text = re.sub( r"/tmp/ipykernel_.*.py",  "/Jupyter/Notebook/Student/Task/Implementation/Cells", tb_text)
        tb_text = re.sub( r"\s{20,}Traceback",  " Traceback", tb_text)
        s_split = tb_text.split('\n')
        if len(s_split) > 0:
            c_s_split = s_split[1:]
            tb_text = '\n'.join(c_s_split) + '\n'
        tb_text = tb_text.replace('\x1b[0;36m', '\x1b[1m \x1b[1;34m')
        return tb_text
except:
    def get_tb_colored_str(exc_type, exc_value, exc_traceback):
        manipulated_exc_traceback = censor_exc_traceback(exc_traceback)
        tb_text = traceback.format_exception(exc_type, exc_value, manipulated_exc_traceback, limit=None, chain=True)    
        tb_text = ''.join(tb_text)
        tb_text = re.sub( r"\"/tmp/ipykernel_.*\"",  "\"/Jupyter/Notebook/Student/Task/Implementation/Cells\"", tb_text)        
        lexer = lexers.get_lexer_by_name("pytb", stripall=True)
        formatter = formatters.get_formatter_by_name("terminal16m")
        tb_colored = highlight(tb_text, lexer, formatter)
        return tb_colored

try:
    from IPython.utils import PyColorize
    color_parser = PyColorize.Parser(color_table=None, out="str", parent=None, style='Linux')
    def code_color_parser(code_str):
        return color_parser.format(code_str)
except:
    def code_color_parser(code_str):
        return code_str

def get_num_indents(src_list):
    assert len(src_list) > 0
    a = [line + 20 * ' ' for line in src_list]
    b = [len(line) - len(line.lstrip()) for line in a]
    assert b[0] == 0
    c = min(b[1:])
    return c

def code_snippet_maker(stu_function, args, kwargs):
    test_kwargs_str_lst = []
    test_kwargs_str_lst.append('from copy import deepcopy')
    test_kwargs_str_lst.append("failed_arguments = deepcopy(test_results['test_kwargs'])")
    for arg_ in args:
        test_kwargs_str_lst.append(arg_)
    for key,val in kwargs.items():
        test_kwargs_str_lst.append(f"{key} = failed_arguments['{key}']") 
    test_kwargs_str = ', '.join(test_kwargs_str_lst)
    
    if hasattr(stu_function, '__name__'):
        stu_func_name = stu_function.__name__
    else:
        stu_func_name = 'YOUR_FUNCTION_NAME'
    
    check_list_code = []
    check_list_code.append(f"correct_sol = test_results['correct_sol'] # The Reference Solution")
    check_list_code.append(f"if isinstance(correct_sol, np.ndarray):")
    check_list_code.append(f"    assert isinstance(my_solution, np.ndarray)")
    check_list_code.append(f"    assert my_solution.dtype is correct_sol.dtype")
    check_list_code.append(f"    assert my_solution.shape == correct_sol.shape")
    check_list_code.append(f"    assert np.allclose(my_solution, correct_sol)")
    check_list_code.append(f"    print('If you passed the above assertions, it probably means that you have fixed the issue! Well Done!')")
    check_list_code.append(f"    print('Now you have to do 3 things:')")
    check_list_code.append(f"    print('  1) Carefully copy the fixed code body back to the {stu_func_name} function.')")
    check_list_code.append(f"    print('  2) If you copied any \"returned_var = \" lines, convert them back to return statements.')")
    check_list_code.append(f"    print('  3) Carefully remove this cell (i.e., the cell you inserted and modified) once you are done.')")

    try:
        src = inspect.getsource(stu_function)
        src_list = src.split('\n')
        src_list = [line for line in src_list if not (line.strip().startswith('#'))]
        no_indents = get_num_indents(src_list)
        mod_src_list = []
        src_gen = src_list[1:] if src_list[0].startswith('def') else src_list
        for line in src_gen:
            if len(line) > no_indents:
                shifted_left_line = line[no_indents:]
            else:
                shifted_left_line = line
            
            return_statement = 'return '
            if not shifted_left_line.lstrip().startswith(return_statement):
                mod_src_list.append(shifted_left_line)
            else:
                i = shifted_left_line.index(return_statement)
                shifted_left_line = shifted_left_line[:i] + 'returned_var = ' + shifted_left_line[i+len(return_statement):] + ' # returned variable'
                mod_src_list.append(shifted_left_line)

        mod_bodysrc_list = '\n'.join(mod_src_list).strip().split('\n')
        
        mod_src_list = []
        mod_src_list = mod_src_list + ['### You can copy the following auto-generated snippet into a new cell to reproduce the issue.']
        mod_src_list = mod_src_list + ['### Use the + button on the top left of the screen to insert a new cell below.']
        mod_src_list = mod_src_list + ['']
        mod_src_list = mod_src_list + ['#'*7 + ' Test Arguments ' + '#'*7] + test_kwargs_str_lst
        mod_src_list = mod_src_list + ['\n' + '#'*7 + ' Your Code Body ' + '#'*7] + mod_bodysrc_list
        mod_src_list.append('\n' + '#'*5 + ' Checking Solutions '+ '#'*6)
        mod_src_list.append(f"my_solution = returned_var # Your Solution")
        mod_src_list = mod_src_list + check_list_code
        processed_code = '\n'.join(mod_src_list)
    except:
        mod_src_list = []
        mod_src_list.append(f"my_solution = {stu_func_name}({test_kwargs_str})")
        mod_src_list = mod_src_list + check_list_code
        processed_code = '\n'.join(mod_src_list)
    
    return processed_code


########################################################################################
####################### Utilities for comparison processing ############################
########################################################################################
def retrieve_item(item_name, ptr_, test_idx, npz_file):
    item_shape = npz_file[f'shape_{item_name}'][test_idx]
    item_size = int(np.prod(item_shape))
    item = npz_file[item_name][ptr_:(ptr_+item_size)].reshape(item_shape)
    return item, ptr_+item_size

class NPStrListCoder:
    def __init__(self):
        self.filler = '?'
        self.spacer = ':'
        self.max_len = 100
    
    def encode(self, str_list):
        my_str_ = self.spacer.join(str_list)
        str_hex_data = [ord(c) for c in my_str_]
        assert_msg = f'Increase max len; you have so many characters: {len(str_hex_data)}>{self.max_len}'
        assert len(str_hex_data) <= self.max_len, assert_msg
        str_hex_data = str_hex_data + [ord(self.filler) for _ in range(self.max_len - len(str_hex_data))]
        str_hex_np = np.array(str_hex_data)
        return str_hex_np
    
    def decode(self, np_arr):
        a = ''.join([chr(i) for i in np_arr])
        recovered_list = a.replace(self.filler, '').split(self.spacer)
        return recovered_list
    
str2np_coder = NPStrListCoder()

def test_case_loader(test_file):
    npz_file = np.load(test_file)
    arg_id_list = sorted([int(key[4:]) for key in npz_file.keys() if key.startswith('arg_')])
    kwarg_names_list = sorted([key[6:] for key in npz_file.keys() if key.startswith('kwarg_')])

    arg_ptr_list = [0 for _ in range(len(arg_id_list))]
    dfcarg_ptr_list = [0 for _ in range(len(arg_id_list))]
    kwarg_ptr_list = [0 for _ in range(len(kwarg_names_list))]
    dfckwarg_ptr_list = [0 for _ in range(len(kwarg_names_list))]
    out_ptr = 0
    for i in np.arange(npz_file['num_tests']):
        args_list = []
        for arg_id, arg_id_ in enumerate(arg_id_list):
            arg_item, arg_ptr_list[arg_id] = retrieve_item(f'arg_{arg_id_}', arg_ptr_list[arg_id], i, npz_file)
            if f'dfcarg_{arg_id_}' in npz_file.keys():
                col_list_code, dfcarg_ptr_list[arg_id] = retrieve_item(f'dfcarg_{arg_id_}', dfcarg_ptr_list[arg_id], i, npz_file)
                arg_item = pd.DataFrame(arg_item, columns=str2np_coder.decode(col_list_code))
            args_list.append(arg_item)
        args = tuple(args_list)

        kwargs = {}
        for kwarg_id, kwarg_name in enumerate(kwarg_names_list):
            kwarg_item, kwarg_ptr_list[kwarg_id] = retrieve_item(f'kwarg_{kwarg_name}', kwarg_ptr_list[kwarg_id], i, npz_file)
            if f'dfckwarg_{kwarg_name}' in npz_file.keys():
                col_list_code, dfckwarg_ptr_list[kwarg_id] = retrieve_item(f'dfckwarg_{kwarg_name}', dfckwarg_ptr_list[kwarg_id], i, npz_file)
                kwarg_item = pd.DataFrame(kwarg_item, columns=str2np_coder.decode(col_list_code))
            kwargs[kwarg_name]=kwarg_item

        output, out_ptr = retrieve_item(f'output', out_ptr, i, npz_file)

        yield args, kwargs, output
        
def arg2str(args, kwargs, adv_user_msg=False, stu_func=None):
    msg = ''
    
    for arg_ in args:
        msg += f'{arg_}\n'
    for key,val in kwargs.items():
        try:
            val_str = np.array_repr(val)
        except:
            val_str = val
        new_line = f'{Fore_MAGENTA}{key}{Fore_BLACK} = {val_str}\n'
        new_line = new_line.replace(' = array(',' = np.array(')
        new_line = new_line.replace('nan,','np.nan,')
        msg += new_line
        
   
    if adv_user_msg:
        try:
            is_stu_func_lambda = isinstance(stu_func, types.LambdaType) 
            if is_stu_func_lambda:
                is_stu_func_lambda = stu_func.__name__ == "<lambda>"
            if not is_stu_func_lambda:
                code_title_ = f'\n' + '-' * (NO_DASHES-1) + f'{Fore_RED} Reproducing Code Snippet {Fore_BLACK}' + '-' * (NO_DASHES-2) + '\n'
                code = code_snippet_maker(stu_func, args, kwargs)
                msg += code_title_ + code_color_parser(code)
        except:
            pass
    return msg


def test_case_checker(stu_func, task_id=0):
    out_dict = {}
    out_dict['task_number'] = task_id
    out_dict['exception'] = None
    out_dict['exception_info'] = None
    test_db_npz = f'{test_db_dir}/task_{task_id}.npz'
    if not os.path.exists(test_db_npz):
        out_dict['message'] = f'Test database test_db/task_{task_id}.npz does not exist... aborting!'
        out_dict['passed'] = False
        out_dict['test_args'] = None
        out_dict['test_kwargs'] = None
        out_dict['stu_sol'] = None
        out_dict['correct_sol'] = None
        return out_dict
    
    if hasattr(stu_func, '__name__'):
        stu_func_name = stu_func.__name__
    else:
        stu_func_name = None
        
    done = False
    err_title = f'\n' + '*' * NO_DASHES + f'{Fore_RED}    Error in Task {task_id}   {Fore_BLACK}' + '*' * NO_DASHES + f'\n'
    test_case_title = '\n' + '-' * NO_DASHES + f'{Fore_RED}  Test Case Arguments  {Fore_BLACK}' + '-' * NO_DASHES + '\n'
    summary_title = '-' * NO_DASHES + f' {Fore_RED}      Summary        {Fore_BLACK}' + '-' * NO_DASHES + '\n'
    for (test_args, test_kwargs, correct_sol) in test_case_loader(test_db_npz):
        try:
            stu_args_copy = copy.deepcopy(test_args)
            stu_kwargs_copy = copy.deepcopy(test_kwargs)
            stu_sol = stu_func(*stu_args_copy, **stu_kwargs_copy)
        except Exception as stu_exception:
            stu_sol = None
            stu_exception_info = sys.exc_info()
            message =  err_title + summary_title
            message +=  f'Your code {Fore_RED}crashed{Fore_BLACK} during the evaluation of a test case argument.' 
            message += f' The rest of this message gives you the following material:\n'
            message += f'  1. The exception traceback detailing how the error occured.\n'
            message += f'  2. The specific test case arguments that caused the error.\n'
            message += f'  3. A code snippet that can conviniently reproduce the error.\n'
            message += f'     -> You can {Fore_RED}copy and paste{Fore_BLACK} the {Fore_RED}code snippet{Fore_BLACK} into a {Fore_RED}new cell{Fore_BLACK}, and run the cell to reproduce the error.\n\n'
            message += '-' * NO_DASHES + f'{Fore_RED}  Exception Traceback  {Fore_BLACK}' + '-' * NO_DASHES + '\n'
            message += get_tb_colored_str(*stu_exception_info)
            message += test_case_title
            message += arg2str(test_args, test_kwargs, adv_user_msg=True, stu_func=stu_func)
            out_dict['test_args'] = test_args
            out_dict['test_kwargs'] = test_kwargs
            out_dict['stu_sol'] = stu_sol
            out_dict['correct_sol'] = correct_sol
            out_dict['message'] = message
            out_dict['passed'] = False
            out_dict['exception'] = stu_exception
            out_dict['exception_info'] = stu_exception_info
            return out_dict
        
        if isinstance(correct_sol, np.ndarray) and np.isscalar(stu_sol):
            # This is handling a special case: When scalar numpy objects are stored, 
            # they will be converted to a numpy array upon loading. 
            # In this case, we'll give students the benefit of the doubt, 
            # and assume the correct solution already was a scalar.
            if correct_sol.size == 1:
                correct_sol = np.float64(correct_sol.item())
                stu_sol = np.float64(np.float64(stu_sol).item())
        
        #Type Sanity check
        if type(stu_sol) is not type(correct_sol):
            message =  err_title + summary_title
            message += f'Your solution\'s {Fore_RED}output type{Fore_BLACK} is not the same as '
            message += f'the reference solution\'s data type.\n' 
            message += f'    Your    solution\'s type --> {Fore_RED}{type(stu_sol)}{Fore_BLACK}\n'
            message += f'    Correct solution\'s type --> {Fore_RED}{type(correct_sol)}{Fore_BLACK}\n'
            message += test_case_title
            message += arg2str(test_args, test_kwargs, adv_user_msg=True, stu_func=stu_func)
            out_dict['test_args'] = test_args
            out_dict['test_kwargs'] = test_kwargs
            out_dict['stu_sol'] = stu_sol
            out_dict['correct_sol'] = correct_sol
            out_dict['message'] = message
            out_dict['passed'] = False
            return out_dict
        
        if isinstance(correct_sol, np.ndarray):
            if not np.all(np.array(correct_sol.shape) == np.array(stu_sol.shape)):
                message =  err_title + summary_title
                message += f'Your solution\'s {Fore_RED}output numpy shape{Fore_BLACK} is not the same as '
                message += f'the reference solution\'s shape.\n'
                message += f'    Your    solution\'s shape --> {Fore_RED}{stu_sol.shape}{Fore_BLACK}\n'
                message += f'    Correct solution\'s shape --> {Fore_RED}{correct_sol.shape}{Fore_BLACK}\n'
                message += test_case_title
                message += arg2str(test_args, test_kwargs, adv_user_msg=True, stu_func=stu_func)
                out_dict['test_args'] = test_args
                out_dict['test_kwargs'] = test_kwargs
                out_dict['stu_sol'] = stu_sol
                out_dict['correct_sol'] = correct_sol
                out_dict['message'] = message
                out_dict['passed'] = False
                return out_dict
            
            if not(stu_sol.dtype is correct_sol.dtype):
                message =  err_title + summary_title
                message += f'Your solution\'s {Fore_RED}output numpy dtype{Fore_BLACK} is not the same as'
                message += f'the reference solution\'s dtype.\n'
                message += f'    Your    solution\'s dtype --> {Fore_RED}np.{stu_sol.dtype}{Fore_BLACK}\n'
                message += f'    Correct solution\'s dtype --> {Fore_RED}np.{correct_sol.dtype}{Fore_BLACK}\n'
                message += test_case_title
                message += arg2str(test_args, test_kwargs, adv_user_msg=True, stu_func=stu_func)
                out_dict['test_args'] = test_args
                out_dict['test_kwargs'] = test_kwargs
                out_dict['stu_sol'] = stu_sol
                out_dict['correct_sol'] = correct_sol
                out_dict['message'] = message
                out_dict['passed'] = False
                return out_dict
        
        if isinstance(correct_sol, np.ndarray):
            equality_array = np.isclose(stu_sol, correct_sol, rtol=1e-05, atol=1e-08, equal_nan=True)
            if not equality_array.all():
                message =  err_title + summary_title
                message += f'Your solution is {Fore_RED}not the same{Fore_BLACK} as the correct solution.\n'
                whr_ = np.array(np.where(np.logical_not(equality_array)))
                ineq_idxs = whr_[:,0].tolist()
                message += f'    your_solution{ineq_idxs}={stu_sol[tuple(ineq_idxs)]}\n'
                message += f'    correct_solution{ineq_idxs}={correct_sol[tuple(ineq_idxs)]}\n'
                message += test_case_title
                message += arg2str(test_args, test_kwargs, adv_user_msg=True, stu_func=stu_func)
                out_dict['test_args'] = test_args
                out_dict['test_kwargs'] = test_kwargs
                out_dict['stu_sol'] = stu_sol
                out_dict['correct_sol'] = correct_sol
                out_dict['message'] = message
                out_dict['passed'] = False
                return out_dict
            
        elif np.isscalar(correct_sol):
            equality_array = np.isclose(stu_sol, correct_sol, rtol=1e-05, atol=1e-08, equal_nan=True)
            if not equality_array.all():
                message =  err_title + summary_title
                message += f'Your solution is {Fore_RED}not the same{Fore_BLACK} as the correct solution.\n'
                message += f'    your_solution={stu_sol}\n'
                message += f'    correct_solution={correct_sol}\n'
                message += test_case_title
                message += arg2str(test_args, test_kwargs, adv_user_msg=True, stu_func=stu_func)
                out_dict['test_args'] = test_args
                out_dict['test_kwargs'] = test_kwargs
                out_dict['stu_sol'] = stu_sol
                out_dict['correct_sol'] = correct_sol
                out_dict['message'] = message
                out_dict['passed'] = False
                return out_dict
        
        elif isinstance(correct_sol, tuple):
            if not correct_sol==stu_sol:
                message =  err_title + summary_title
                message += f'Your solution is {Fore_RED}not the same{Fore_BLACK} as the correct solution.\n'
                message += f'    your_solution={stu_sol}\n'
                message += f'    correct_solution={correct_sol}\n'
                message += test_case_title
                message += arg2str(test_args, test_kwargs, adv_user_msg=True, stu_func=stu_func)
                out_dict['test_args'] = test_args
                out_dict['test_kwargs'] = test_kwargs
                out_dict['stu_sol'] = stu_sol
                out_dict['correct_sol'] = correct_sol
                out_dict['message'] = message
                out_dict['passed'] = False
                return out_dict
        
        else:
            raise Exception(f'Not implemented comparison for other data types. sorry!')
        
    out_dict['test_args'] = None
    out_dict['test_kwargs'] = None
    out_dict['stu_sol'] = None
    out_dict['correct_sol'] = None
    out_dict['message'] = 'Well Done!'
    out_dict['passed'] = True
    return out_dict

def show_test_cases(test_func, task_id=0):
    from IPython.display import clear_output
    file_path = f'{test_db_dir}/task_{task_id}.npz'
    npz_file = np.load(file_path)
    orig_images = npz_file['raw_images']
    ref_images = npz_file['ref_images']
    test_images = test_func(orig_images)
    
    visualize_ = visualize and perform_computation
    
    if not np.all(np.array(test_images.shape) == np.array(ref_images.shape)):
        print(f'Error: It seems the test images and the ref images have different shapes. Modify your function so that they both have the same shape.')
        print(f' test_images shape: {test_images.shape}')
        print(f' ref_images shape: {ref_images.shape}')
        return None, None, None, False
    
    if not np.all(np.array(test_images.dtype) == np.array(ref_images.dtype)):
        print(f'Error: It seems the test images and the ref images have different dtype. Modify your function so that they both have the same dtype.')
        print(f' test_images dtype: {test_images.dtype}')
        print(f' ref_images dtype: {ref_images.dtype}')
        return None, None, None, False
    
    for i in range(ref_images.shape[0]):
        if visualize_:
            nrows, ncols = 1, 3
            ax_w, ax_h = 5, 5
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*ax_w, nrows*ax_h))
            axes = np.array(axes).reshape(nrows, ncols)
        
        orig_image = orig_images[i]
        ref_image = ref_images[i]
        test_image = test_images[i]
        
        if visualize_:
            ax = axes[0,0]
            ax.pcolormesh(orig_image, edgecolors='k', linewidth=0.01, cmap='Greys')
            ax.xaxis.tick_top()
            ax.invert_yaxis()
            
            x_ticks = ax.get_xticks(minor=False).astype(np.int)
            x_ticks = x_ticks[x_ticks < orig_image.shape[1]]
            ax.set_xticks(x_ticks + 0.5)
            ax.set_xticklabels((x_ticks).astype(np.int))
            
            y_ticks = ax.get_yticks(minor=False).astype(np.int)
            y_ticks = y_ticks[y_ticks < orig_image.shape[0]]
            ax.set_yticks(y_ticks + 0.5)
            ax.set_yticklabels((y_ticks).astype(np.int))
            
            ax.set_aspect('equal')
            ax.set_title('Raw Image')

            ax = axes[0,1]
            ax.pcolormesh(ref_image, edgecolors='k', linewidth=0.01, cmap='Greys')
            ax.xaxis.tick_top()
            ax.invert_yaxis()
            
            x_ticks = ax.get_xticks(minor=False).astype(np.int)
            x_ticks = x_ticks[x_ticks < ref_image.shape[1]]
            ax.set_xticks(x_ticks+0.5)
            ax.set_xticklabels((x_ticks).astype(np.int))
            
            y_ticks = ax.get_yticks(minor=False).astype(np.int)
            y_ticks = y_ticks[y_ticks < ref_image.shape[0]]
            ax.set_yticks(y_ticks+0.5)
            ax.set_yticklabels((y_ticks).astype(np.int))
            
            ax.set_aspect('equal')
            ax.set_title('Reference Solution Image')

            ax = axes[0,2]
            ax.pcolormesh(test_image, edgecolors='k', linewidth=0.01, cmap='Greys')
            ax.xaxis.tick_top()
            ax.invert_yaxis()
            
            x_ticks = ax.get_xticks(minor=False).astype(np.int)
            x_ticks = x_ticks[x_ticks < test_image.shape[1]]
            ax.set_xticks(x_ticks + 0.5)
            ax.set_xticklabels((x_ticks).astype(np.int))
            
            y_ticks = ax.get_yticks(minor=False).astype(np.int)
            y_ticks = y_ticks[y_ticks < test_image.shape[0]]
            ax.set_yticks(y_ticks + 0.5)
            ax.set_yticklabels((y_ticks).astype(np.int))
            
            ax.set_aspect('equal')
            ax.set_title('Your Solution Image')
        
        if np.allclose(ref_image, test_image):
            if visualize_:
                print('The reference and solution images are the same to a T! Well done on this test case.')
        else:
            print('The reference and solution images are not the same...')
            ineq_idxs = np.array(np.where(np.logical_not(np.isclose(ref_image, test_image))))[:,0].tolist()
            print(f'ref_image{ineq_idxs}={ref_image[tuple(ineq_idxs)]}')
            print(f'test_image{ineq_idxs}={test_image[tuple(ineq_idxs)]}')
            if visualize_:
                print('I will return the images so that you will be able to diagnose the issue and resolve it...')
            return (orig_image, ref_image, test_image, False)
            
        if visualize_:
            plt.show()
            input_prompt = '    Enter nothing to go to the next image\nor\n    Enter "s" when you are done to recieve the three images. \n'
            input_prompt += '        **Don\'t forget to do this before continuing to the next step.**\n'
            
            try:
                cmd = input(input_prompt)
            except KeyboardInterrupt:
                cmd = 's'
            
            if cmd.lower().startswith('s'):
                return (orig_image, ref_image, test_image, True)
            else:
                clear_output(wait=True)
        
    return (orig_image, ref_image, test_image, True)