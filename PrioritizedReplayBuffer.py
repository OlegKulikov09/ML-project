import numpy as np  # Убедитесь, что NumPy импортирован


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []

    def add(self, transition, error):
        """Добавляет переход в буфер"""
        self.buffer.append(transition)
        self.priorities.append((abs(error) + 1e-5) ** self.alpha)  # Приоритет с ε-гарантией
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
            self.priorities.pop(0)

    def sample(self, batch_size):
        """Возвращает выборку из буфера с учетом приоритетов"""
        priorities = np.array(self.priorities, dtype=np.float32)
        scaled_priorities = priorities / (priorities.sum() + 1e-8)  # Нормализация
        if not np.isclose(scaled_priorities.sum(), 1.0):
            raise ValueError("Приоритеты не нормализованы должным образом.")
        indices = np.random.choice(len(self.buffer), batch_size, p=scaled_priorities)
        batch = [self.buffer[idx] for idx in indices]
        return batch, indices

    def update_priorities(self, indices, errors):
        """Обновляет приоритеты для выбранных переходов"""
        for i, error in zip(indices, errors):
            self.priorities[i] = (abs(error) + 1e-5) ** self.alpha
