from modules.identifiers.dict_keys import DictKeys
from modules.execution.stopppable_thread import StoppableThread
from queue import Empty
import time


class QueueReader(DictKeys):
    def __init__(self, main_app):
        super().__init__()
        self.main_app = main_app
        self.queue_reader_thread = None
        self.queue = main_app.queue

    def start(self):
        self.queue_reader_thread = StoppableThread(target=self.read_process_queue)
        self.queue_reader_thread.start()

    def stop(self):
        self.queue_reader_thread.stop()

    def read_process_queue(self):
        while True:
            time.sleep(0.0001)
            if self.queue_reader_thread.stopped():
                break
            try:
                output = self.queue.get_nowait()
            except Empty:
                continue
            # gotta find another way to do this if too many if ... maybe do another map thing with a dict. dunno
            if 'dbo_output' in str(output):
                self.main_app.update_dbo_info_frame(output)
                continue

            if 'dbo_job_progress' in str(output):
                self.main_app.update_job_progress(output)
                continue

            if isinstance(output, type(str())):
                print(output)
            else:
                try:
                    self.main_app.output_handlers[output.process_name].main(output)
                    self.main_app.update_info_frame(output)
                except KeyError:
                    print(str(output))
