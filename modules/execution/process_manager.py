class ProcessManager:
    def __init__(self):
        self.running_processes = []
        self.selected_process = None

    def set_selected_process(self, process_name):
        for process in self.running_processes:
            if str(process) == process_name:
                self.selected_process = process
                break

    def stop_selected_process(self):
        for process in self.running_processes:
            if process == self.selected_process:
                process.terminate()
                self.running_processes.remove(process)
                break

    def stop_all_processes(self):
        while self.running_processes:
            for process in self.running_processes:
                process.terminate()
                self.running_processes.remove(process)
