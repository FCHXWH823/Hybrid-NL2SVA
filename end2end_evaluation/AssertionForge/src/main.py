# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#!/opt/conda/bin/python

from gen_plan import gen_plan
from gen_KG_graphRAG import build_KG
from use_KG import use_KG

from saver import saver
from config import FLAGS
from utils import OurTimer, get_root_path, report_save_dir
import traceback
import sys

sys.path.insert(0, '../')
# sys.path.insert(0, '../../../../fv_eval')


timer = OurTimer()


def main():
    if FLAGS.task == 'gen_plan':
        gen_plan()
    elif FLAGS.task == 'build_KG':
        build_KG()
    elif FLAGS.task == 'use_KG':
        use_KG()
    else:
        raise NotImplementedError()


if __name__ == '__main__':

    timer = OurTimer()

    try:
        main()
        status = 'Complete'
    except:
        traceback.print_exc()
        s = '\n'.join(traceback.format_exc(limit=-1).split('\n')[1:])
        saver.log_info(traceback.format_exc(), silent=True)
        saver.save_exception_msg(traceback.format_exc())
        status = 'Error'

    tot_time = timer.time_and_clear()
    saver.log_info(f'Total time: {tot_time}; {report_save_dir(saver.get_log_dir())}')
    saver.close()
