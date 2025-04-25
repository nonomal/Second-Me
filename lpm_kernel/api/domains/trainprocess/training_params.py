class TrainingParams:
    """
    Bean class to encapsulate training parameters used in both start_process and retrain.
    """
    def __init__(
        self,
        model_name=None,
        learning_rate=None,
        number_of_epochs=None,
        concurrency_threads=None,
        data_synthesis_mode=None,
        use_cuda=False,
        is_cot=False
    ):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.number_of_epochs = number_of_epochs
        self.concurrency_threads = concurrency_threads
        self.data_synthesis_mode = data_synthesis_mode
        self.use_cuda = use_cuda
        self.is_cot = is_cot
    
    @classmethod
    def from_request(cls, request_data):
        """
        Create a TrainingParams instance from request data
        """
        if not request_data:
            return cls()
            
        return cls(
            model_name=request_data.get("model_name"),
            learning_rate=request_data.get("learning_rate"),
            number_of_epochs=request_data.get("number_of_epochs"),
            concurrency_threads=request_data.get("concurrency_threads"),
            data_synthesis_mode=request_data.get("data_synthesis_mode"),
            use_cuda=request_data.get("use_cuda", False),
            is_cot=request_data.get("is_cot")
        )
    
    def to_dict(self):
        """
        Convert the bean to dictionary
        """
        return {
            "model_name": self.model_name,
            "learning_rate": self.learning_rate,
            "number_of_epochs": self.number_of_epochs,
            "concurrency_threads": self.concurrency_threads,
            "data_synthesis_mode": self.data_synthesis_mode,
            "use_cuda": self.use_cuda,
            "is_cot": self.is_cot
        }
    
    def validate(self):
        """
        Validate required parameters
        """
        if not self.model_name:
            return False
        return True
