import carla
import random
import time

def main():
    # 1. CARLA 서버 연결
    client = carla.Client('172.28.224.1', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    
    vehicle = None
    camera = None
    
    try:
        # 2. 차량 스폰 (테슬라 모델3)
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        
        vehicle.set_autopilot(True)
        print("✅ 차량 스폰 및 오토파일럿 활성화 완료")
        
        # 3. RGB 카메라 부착 (차량 루프 위)
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        
        print("👀 Windows의 CARLA 서버 창을 확인하세요! 차량 시점으로 이동합니다.")
        
        # 4. 시각적 검증: 서버의 관찰자(Spectator) 시점을 차량 카메라 위치로 지속적 동기화
        spectator = world.get_spectator()
        
        # 15초간 유지 (time.sleep(15) 대신 0.1초씩 150번 루프를 돌며 시점 업데이트)
        for _ in range(150):
            # 차량에 부착된 카메라의 현재 위치를 가져와서 서버 화면에 적용
            spectator.set_transform(camera.get_transform())
            time.sleep(0.1)
            
    finally:
        # 5. 자원 정리
        print("🧹 리소스 정리 중...")
        if camera is not None:
            camera.destroy()
        if vehicle is not None:
            vehicle.destroy()
        print("종료 완료.")

if __name__ == '__main__':
    main()