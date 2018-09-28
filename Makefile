OBJS = build/entropy_test.o build/max_gain_demo_test.o
TEST_OBJS =  build/entropy_test.o build/max_gain_demo_test.o
CC = g++ -std=c++11 -O3
DEBUG = -g  
CFLAGS = -c $(DEBUG)
LFLAGS = $(DEBUG)

active_VaR_test: build/activeVaR.o
	$(CC) $(LFLAGS) -pg build/activeVaR.o  -o active_VaR
	
	
build/activeVaR.o: src/activeVaR.cpp include/mdp.hpp include/confidence_bounds.hpp include/grid_domain.hpp include/unit_norm_sampling.hpp include/feature_gain.hpp include/feature_birl.hpp
	$(CC) $(CFLAGS) -pg src/activeVaR.cpp -o build/activeVaR.o
		
clean:
	\rm build/*.o src/*~ active_VaR

