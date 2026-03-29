def interpolate_trapezoidal(start_q, end_q, t, duration):
    """
    Nội suy hình thang đơn giản (Vận tốc không đổi ở giữa)
    t: thời gian hiện tại kể từ lúc bắt đầu di chuyển
    duration: tổng thời gian di chuyển mong muốn
    """
    if t >= duration:
        return end_q
    
    # Tỷ lệ hoàn thành (0.0 đến 1.0)
    s = t / duration
    
    # Công thức nội suy mịn (Smoothstep - thay cho hình thang để code gọn)
    # s = 3*s^2 - 2*s^3
    s_smooth = s * s * (3 - 2 * s)
    
    return start_q + (end_q - start_q) * s_smooth