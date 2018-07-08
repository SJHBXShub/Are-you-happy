import runner

def demo():
    config_fp = '../conf/featwheel.conf'
    cv_exrc = runner.SingleExec(config_fp)
    cv_exrc.run_offline()
    #cv_exrc.run_online()
if __name__ == '__main__':
    demo()