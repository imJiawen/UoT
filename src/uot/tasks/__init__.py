def get_task(args):
    if args.task == '20q':
        from src.uot.tasks.twenty_question import Q20Task
        return Q20Task(args)
    elif args.task == 'md':
        from src.uot.tasks.medical_diagnosis import MDTask
        return MDTask(args)
    elif args.task == 'tb':
        from src.uot.tasks.troubleshooting import TBTask
        return TBTask(args)
    elif args.task == 'mediq':
        from src.uot.tasks.mediq import MediQTask
        return MediQTask(args)
    else:
        raise NotImplementedError
