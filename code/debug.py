import runner

def demo():
    config_fp = '../conf/featwheel.conf'
    sing_exrc = runner.SingleExec(config_fp)
    sing_exrc.run_offline()
    sing_exrc.run_online()
if __name__ == '__main__':
    demo()